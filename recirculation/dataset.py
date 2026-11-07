from pathlib import Path
from urllib.request import urlopen

import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def pg19_raw_documents(name, revision, split, start_document_idx, document_limit):
    """Yield original PG-19 books without normalizing their CRLF line endings."""
    if name != "deepmind/pg19" or split != "test":
        raise ValueError("Raw PG-19 evaluation requires deepmind/pg19, test split")

    manifest = hf_hub_download(
        repo_id=name, repo_type="dataset", revision=revision,
        filename="data/test_files.txt",
    )
    filenames = sorted(Path(manifest).read_text(encoding="utf-8").splitlines())
    for filename in filenames[start_document_idx:document_limit]:
        with urlopen(f"https://storage.googleapis.com/deepmind-gutenberg/{filename}", timeout=60) as response:
            yield {"text": response.read().decode("utf-8")}

def dataloader(
        tokenizer,
        start_document_idx=0,
        start_window_idx=0,
        name="emozilla/pg19",
        split="test",
        revision="c021754c8e01c5b1cc83a1f549c1f97fbbb756b8",
        window_size=1023,
        window_prefix="BOS",
        document_limit=None,
        source="huggingface",
        ):
    """Yield independently scored windows from PG-19 documents.

    "BOS" prepends BOS to every window and retains final fragments with at
    least 32 text tokens, padding them on the right. "first_only" prepends BOS
    once per document, splits into full windows, and discards any final fragment.
    Both policies use window_size + 1 positions; the paper does not specify
    the exact BOS-plus-text accounting for its BOS-present windows.

    Args:
        tokenizer: Tokenizer used to encode document text.
        start_document_idx: Zero-based document index at which to resume.
        start_window_idx: Zero-based window index within the starting document;
            later documents start at window zero.
        name: Dataset repository name.
        split: Dataset split to stream.
        revision: Dataset revision to load.
        window_size: Text tokens per BOS-prefixed window; all windows have
            window_size + 1 positions.
        window_prefix: "BOS" for BOS in every window, or "first_only" for
            BOS only in the first window of each document.
        document_limit: Cap on documents from the start of the split, applied
            before skipping to start_document_idx; None selects the full split.
        source: "huggingface" for the dataset loader, or "raw_pg19" to preserve
            CRLF from the original DeepMind test files.

    Yields:
        (window_ids, window_mask, document_idx, window_idx), where the first two
        are 1-D CPU tensors of length window_size + 1 and the indices are
        zero-based. The mask is 1 for present tokens, 0 for padding.
    """
    if window_prefix not in ("BOS", "first_only"):
        raise ValueError(f"Unknown window_prefix: {window_prefix}")

    if source == "raw_pg19":
        dataset = pg19_raw_documents(name, revision, split, start_document_idx, document_limit)
    elif source == "huggingface":
        dataset = load_dataset(
            name,
            split=split,
            streaming=True,
            revision=revision,
        )
        if document_limit is not None:
            dataset = dataset.take(document_limit)
        dataset = dataset.skip(start_document_idx)
    else:
        raise ValueError(f"Unknown document source: {source}")

    for document_idx, document in enumerate(dataset, start=start_document_idx):

        inputs = tokenizer(
            document["text"],
            return_tensors="pt",
            add_special_tokens=False
        )

        first_window = start_window_idx if document_idx == start_document_idx else 0

        if window_prefix == "first_only":
            # The first window starts with BOS; later ones are text-only.
            window_length = window_size + 1
            sequence = F.pad(inputs["input_ids"].squeeze(0), (1, 0), value=tokenizer.bos_token_id)
            sequence_mask = F.pad(inputs["attention_mask"].squeeze(0), (1, 0), value=1)
            for window_idx in range(first_window, sequence.numel() // window_length):
                start = window_idx * window_length
                yield (
                    sequence[start:start + window_length],
                    sequence_mask[start:start + window_length],
                    document_idx,
                    window_idx,
                )
            continue

        total_tokens = inputs["input_ids"].shape[-1]
        num_of_full_windows = total_tokens // window_size

        for window_idx in range(first_window, num_of_full_windows):
            window_ids = F.pad(inputs["input_ids"][:, window_size * window_idx : window_size * (window_idx + 1)].squeeze(0), (1, 0), value=tokenizer.bos_token_id)
            window_mask = F.pad(inputs["attention_mask"][:, window_size * window_idx : window_size * (window_idx + 1)].squeeze(0), (1, 0), value=1)

            yield window_ids, window_mask, document_idx, window_idx

        if total_tokens - num_of_full_windows * window_size >= 32 and num_of_full_windows >= first_window:
            window_ids = inputs["input_ids"][:, window_size * num_of_full_windows : ].squeeze(0)
            window_ids = F.pad(window_ids, (1, 0), value=tokenizer.bos_token_id)
            leftover_tokens = window_ids.shape[-1]
            window_ids = F.pad(window_ids, (0, (window_size + 1) - leftover_tokens), value=tokenizer.pad_token_id)

            window_mask = inputs["attention_mask"][:, window_size * num_of_full_windows : ].squeeze(0)
            window_mask = F.pad(window_mask, (1, 0), value=1)
            window_mask = F.pad(window_mask, (0, (window_size + 1) - leftover_tokens), value=0)

            yield window_ids, window_mask, document_idx, num_of_full_windows

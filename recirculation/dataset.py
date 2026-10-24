import torch.nn.functional as F
from datasets import load_dataset

def dataloader(tokenizer, split="test", window_size=1023, document_limit=None):
    """Yield independent, BOS-prefixed windows from the PG19 dataset.

    Each window contains up to `window_size` text tokens plus one BOS token.
    This treats the paper's 1024-position context as BOS + 1023 text tokens;
    the paper does not specify that packing detail. Final fragments with fewer
    than 32 text tokens are discarded; retained fragments are right-padded.

    Args:
        tokenizer: Tokenizer used to encode documents.
        window_size: Maximum text tokens per window, excluding BOS.
        document_limit: Number of first test documents to process, or None
            for the full split.

    Yields:
        A pair of 1-D CPU tensors (input IDs, attention mask), each of length
        `window_size + 1`. The mask is 1 for BOS and text, 0 for padding.
    """
    dataset = load_dataset(
        "emozilla/pg19",
        split=split,
        streaming=True,
        revision="c021754c8e01c5b1cc83a1f549c1f97fbbb756b8", # pinned revision has not been verified as the paper's exact source
    )

    if document_limit is not None:
        dataset = dataset.take(document_limit)

    for document_idx, document in enumerate(dataset):

        inputs = tokenizer(
            document["text"],
            return_tensors="pt",
            add_special_tokens=False
        )

        total_tokens = inputs["input_ids"].shape[-1]
        num_of_full_windows = total_tokens // window_size

        for window_idx in range(num_of_full_windows):
            window_ids = F.pad(inputs["input_ids"][:, window_size * window_idx : window_size * (window_idx + 1)].squeeze(0), (1, 0), value=tokenizer.bos_token_id)
            window_mask = F.pad(inputs["attention_mask"][:, window_size * window_idx : window_size * (window_idx + 1)].squeeze(0), (1, 0), value=1)

            yield window_ids, window_mask, document_idx, window_idx

        if total_tokens - num_of_full_windows * window_size >= 32:
            window_ids = inputs["input_ids"][:, window_size * num_of_full_windows : ].squeeze(0)
            window_ids = F.pad(window_ids, (1, 0), value=tokenizer.bos_token_id)
            leftover_tokens = window_ids.shape[-1]
            window_ids = F.pad(window_ids, (0, (window_size + 1) - leftover_tokens), value=tokenizer.pad_token_id)

            window_mask = inputs["attention_mask"][:, window_size * num_of_full_windows : ].squeeze(0)
            window_mask = F.pad(window_mask, (1, 0), value=1)
            window_mask = F.pad(window_mask, (0, (window_size + 1) - leftover_tokens), value=0)

            yield window_ids, window_mask, document_idx, num_of_full_windows

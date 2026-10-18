from datasets import load_dataset
import torch.nn.functional as F

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

    for sample in dataset:

        inputs = tokenizer(
            sample["text"],
            return_tensors="pt",
            add_special_tokens=False
        )

        total_tokens = inputs["input_ids"].shape[-1]
        num_of_full_windows = total_tokens // window_size

        for idx in range(num_of_full_windows):
            window_ids = F.pad(inputs["input_ids"][:, window_size * idx : window_size * (idx + 1)].squeeze(0), (1, 0), value=tokenizer.bos_token_id)
            window_mask = F.pad(inputs["attention_mask"][:, window_size * idx : window_size * (idx + 1)].squeeze(0), (1, 0), value=1)

            yield window_ids, window_mask

        if total_tokens - num_of_full_windows * window_size >= 32:
            window_ids = inputs["input_ids"][:, window_size * num_of_full_windows : ].squeeze(0)
            window_ids = F.pad(window_ids, (1, 0), value=tokenizer.bos_token_id)
            leftover_tokens = window_ids.shape[-1]
            window_ids = F.pad(window_ids, (0, (window_size + 1) - leftover_tokens), value=tokenizer.pad_token_id)

            window_mask = inputs["attention_mask"][:, window_size * num_of_full_windows : ].squeeze(0)
            window_mask = F.pad(window_mask, (1, 0), value=1)
            window_mask = F.pad(window_mask, (0, (window_size + 1) - leftover_tokens), value=0)

            yield window_ids, window_mask

def evaluate_window(model, window_ids, window_mask, vanilla=False):
    # Add batch dimension and move to model device
    window_ids = window_ids.unsqueeze(0).to(model.model.device)
    window_mask = window_mask.unsqueeze(0).to(model.model.device)

    if vanilla:
        logits, targets = model.vanilla_run(window_ids, window_mask)
    else:
        logits, targets = model.recirculated_run(window_ids, window_mask)

    logits = logits[:, :-1, :].float()  # The final logits are predictions for the next token ID
    targets = targets[:, 1:]            # The token IDs from the next token are targets of the previous final logits
    mask = window_mask[:, 1:]           # The mask needs to be shifted accordingly

    log_probs = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(
        log_probs.reshape(-1, log_probs.size(-1)),      # Expects [N, C]
        targets.reshape(-1),                            # Expects [N]
        reduction="none"
    )

    valid_nll = nll[mask.reshape(-1).bool()]    # Mask PAD tokens

    return valid_nll.sum().item(), valid_nll.numel()

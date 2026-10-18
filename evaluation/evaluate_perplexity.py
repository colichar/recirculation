import math
import torch.nn.functional as F
from recirculation.dataset import dataloader

MODES = ["vanilla", "recirculated"]


def score_window_nll(model, window_ids, window_mask, mode="vanilla"):
    """Score valid next-token targets in one BOS-prefixed window.

    Run vanilla or Recirculation inference on the same input IDs and attention
    mask. Logits at each position predict the following token; exclude BOS as
    a target and exclude padded targets from the NLL sum. Log probabilities
    are computed in FP32.

    Args:
        model: RecirculatedGemma instance used for inference.
        window_ids: 1-D token IDs for one window, initially on CPU.
        window_mask: 1-D mask with 1 for BOS and text, 0 for padding.
        vanilla: Use the vanilla forward pass instead of Recirculation.

    Returns:
        A pair (summed NLL, scored-token count) for valid next-token targets.
    """

    # Add batch dimension and move to model device
    window_ids = window_ids.unsqueeze(0).to(model.model.device)
    window_mask = window_mask.unsqueeze(0).to(model.model.device)

    if mode=="vanilla":
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

def evaluate_perplexity(model, document_limit=None):
    """Evaluate vanilla and Recirculation perplexity on paired PG-19 windows.

    Stream each independent BOS-prefixed window once and score both modes on
    the same input IDs and valid target positions. For each mode, compute
    perplexity from its summed NLL divided by its total scored-token count.
    Raise ValueError if the modes have different total scored-token counts.

    Args:
        model: RecirculatedGemma instance, including its tokenizer and
            Recirculation configuration.
        document_limit: Number of first test documents to evaluate, or None
            for the full test split.

    Returns:
        A tuple (perplexity_by_mode, token_counts_by_mode, window_count).
        The first two values are dictionaries keyed by "vanilla" and
        "recirculated"; window_count is shared by both modes.
    """
    total_nll_sum = {mode: 0 for mode in MODES}
    total_tokens = {mode: 0 for mode in MODES}
    total_windows = 0
    
    for window_ids, window_mask in dataloader(tokenizer=model.tokenizer, document_limit=document_limit):
        for mode in MODES:
            nll_sum, num_tokens = score_window_nll(
                model=model,
                window_ids=window_ids,
                window_mask=window_mask,
                mode=mode
            )
            total_nll_sum[mode] += nll_sum
            total_tokens[mode] += num_tokens
        total_windows += 1

    if total_tokens["vanilla"] != total_tokens["recirculated"]:
        raise ValueError(f"Scored-token counts differ: {total_tokens}")

    perplexity = {mode: math.exp(total_nll_sum[mode] / total_tokens[mode]) for mode in MODES}

    return perplexity, total_tokens, total_windows


if __name__ == "__main__":
    import torch
    from recirculation.recirculated_gemma import RecirculatedGemma
    
    model = RecirculatedGemma(
        source=10,
        destination=4,
        alpha=0.07,
        alpha_ramp=True,
        model_name="google/gemma-3-1b-pt",
        dtype=torch.float32
    )

    perplexity, total_tokens, total_windows = evaluate_perplexity(
        model=model,
        document_limit=1
    )

    print(f"Evaluation ran for {total_windows} windows and {total_tokens["vanilla"]} tokens\n")
    for mode in MODES:
        print(f"Perplexity {mode} \t: {perplexity[mode]}")
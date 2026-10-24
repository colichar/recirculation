import os
import math
import torch.nn.functional as F
import json
from pathlib import Path
from tqdm.auto import tqdm
from datetime  import datetime, timezone
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

def evaluate_perplexity(
        model,
        dataloader_config,
        checkpoint = None,
        document_limit=None,
        window_limit=None,
        on_window=None
        ):
    """Evaluate vanilla and Recirculation perplexity on paired PG-19 windows.

    Score both modes on the same BOS-prefixed windows and valid target positions.
    When resuming, continue after the checkpoint's last completed window and add
    new scores to its cumulative totals.

    Args:
        model: RecirculatedGemma instance, including its tokenizer.
        dataloader_config: Keyword arguments for dataloader, excluding tokenizer,
            resume indices, and document_limit.
        checkpoint: Saved checkpoint object with a last-window cursor and
            cumulative results, or None for a fresh run. Its NLL and token-count
            dictionaries are updated in place.
        document_limit: Cap on documents from the start of the split, or None
            for the full split.
        window_limit: Maximum additional windows to score in this invocation,
            or None to score all remaining selected windows.
        on_window: Optional callback called after both modes complete a window.
            Receives (document_idx, window_idx, total_nll_sum, total_tokens,
            total_documents, total_windows).

    Returns:
        A seven-item tuple containing, in order:
            perplexity: Dict mapping "vanilla" and "recirculated" to perplexity
                computed from cumulative NLL and scored-token counts.
            total_nll_sum: Dict of cumulative summed NLL for each mode.
            total_tokens: Dict of cumulative scored-token counts for each mode.
            total_documents: Cumulative number of documents with at least one
                scored window.
            total_windows: Cumulative number of scored windows.
            document_idx: Zero-based index of the last completed document.
            window_idx: Zero-based index of the last completed window within
                that document.

    Raises:
        ValueError: If the modes have different scored-token counts or no
            windows have been scored in total.
    """
    if checkpoint is None:
        total_nll_sum = {mode: 0 for mode in MODES}
        total_tokens = {mode: 0 for mode in MODES}
        total_windows = 0
        total_documents = 0
        start_windows = 0
        start_window_idx = 0
        start_document_idx = 0
        last_scored_document_idx = None
    else:
        total_nll_sum = checkpoint["results"]["total_nll_sum"]
        total_tokens = checkpoint["results"]["total_tokens"]
        total_windows = checkpoint["results"]["total_windows"]
        total_documents = checkpoint["results"]["total_documents"]
        start_windows = checkpoint["results"]["total_windows"]

        document_idx = checkpoint["last_document_idx"]
        window_idx = checkpoint["last_window_idx"]
        start_document_idx = document_idx
        start_window_idx = window_idx + 1
        last_scored_document_idx = checkpoint["last_document_idx"]


    for window_ids, window_mask, document_idx, window_idx in tqdm(
        dataloader(
            tokenizer=model.tokenizer,
            start_document_idx=start_document_idx,
            start_window_idx=start_window_idx,
            document_limit=document_limit,
            **dataloader_config
            ),
        desc="PG-19 · vanilla + recirc",
        unit=" window",
        dynamic_ncols=True,
        ):
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
        if document_idx != last_scored_document_idx:
            total_documents += 1
            last_scored_document_idx = document_idx
        if on_window is not None:
            on_window(document_idx, window_idx, total_nll_sum, total_tokens, total_documents, total_windows)
        if window_limit is not None and total_windows == window_limit + start_windows:
            print(f"Reached {window_limit} windows, stopping evaluation at document {document_idx}, window {window_idx}")
            break

    if total_tokens["vanilla"] != total_tokens["recirculated"]:
        raise ValueError(f"Scored-token counts differ: {total_tokens}")
    if total_windows == 0:
        raise ValueError("No windows were scored")

    perplexity = {mode: math.exp(total_nll_sum[mode] / total_tokens[mode]) for mode in MODES}

    return perplexity, total_nll_sum, total_tokens, total_documents, total_windows, document_idx, window_idx

def save_run_metadata(metadata, output):
    """Atomically replace a run's JSON file with its current metadata.

    Write to a temporary file beside the output, then replace the output so an
    interrupted write does not leave a partially written checkpoint.

    Args:
        metadata: JSON-serializable run metadata and checkpoint.
        output: Path of the run's JSON file.
    """
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    # Load Config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="JSON experiment configuration")
    parser.add_argument("--resume-from", type=Path, help="Saved run JSON")
    parser.add_argument("--document-limit", type=int, default=None, help="Limits the full dataset to the first document_limit documents")
    parser.add_argument("--window-limit", type=int, help="Limits the evaluation run to window_limit windows")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataloader_config = config["dataloader"]
    model_config = config["model_parameters"]

    previous_run = None

    if args.resume_from is not None:
        previous_run = json.loads(args.resume_from.read_text(encoding="utf-8"))
        for section in ("dataloader", "model_parameters"):
            if previous_run[section] != config[section]:
                raise ValueError(f"Cannot resume: {section} differs from the saved run")

    checkpoint = previous_run["checkpoint"] if previous_run is not None else None

    document_limit = args.document_limit
    window_limit = args.window_limit

    # Load model
    import torch
    from recirculation.recirculated_gemma import RecirculatedGemma

    started_at = datetime.now(timezone.utc).isoformat()

    dtype_by_name = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    model_kwargs = model_config.copy()
    model_kwargs["dtype"] = dtype_by_name[model_kwargs["dtype"]]
    model = RecirculatedGemma(**model_kwargs)

    import subprocess

    metadata={
        "meta": {
            "evaluation": "perplexity",
            "started_at": started_at,
            "finished_at": None
        },
        "dataloader": dataloader_config,
        "model_parameters": model_config,
        "window_limit": window_limit,
        "document_limit": document_limit,
        "checkpoint": None,
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()),
            "beta": model.beta,
            "alpha_ramp_steps": 10 if model.alpha_ramp else None
        }
    }

    dataset_name = dataloader_config["name"].split("/")[-1]
    model_name = model_config["model_name"].split("/")[-1]
    output = Path("results") / (
        f"{dataset_name}-{model_name}-{started_at.replace(':', '-')}.json"
    )

    def record_window(document_idx, window_idx, total_nll_sum, total_tokens, total_documents, total_windows):
        if total_tokens["vanilla"] != total_tokens["recirculated"]:
            raise ValueError(f"Scored-token counts differ: {total_tokens}")

        metadata["checkpoint"] = {
            "last_document_idx": document_idx,
            "last_window_idx": window_idx,
            "results": {
                "perplexity": {
                    mode: math.exp(total_nll_sum[mode] / total_tokens[mode])
                    for mode in MODES
                },
                "total_nll_sum": total_nll_sum.copy(),
                "total_tokens": total_tokens.copy(),
                "total_windows": total_windows,
                "total_documents": total_documents,
            },
        }
        save_run_metadata(metadata, output)


    # Run evaluation
    dataloader_kwargs = dataloader_config.copy()
    dataloader_kwargs.pop("window_prefix", None) # window_prefix is currently an unused parameter
    perplexity, total_nll_sum, total_tokens, total_documents, total_windows, document_idx, window_idx = evaluate_perplexity(
        model=model,
        dataloader_config=dataloader_kwargs,
        checkpoint=checkpoint,
        document_limit=document_limit,
        window_limit=window_limit,
        on_window=record_window
    )

    # Save metadata
    if metadata["checkpoint"] is None:
        metadata["checkpoint"] = checkpoint

    metadata["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()

    save_run_metadata(metadata, output)

    print(f"Evaluation ran for {total_windows} windows and {total_tokens["vanilla"]} tokens\n")
    for mode in MODES:
        print(f"Perplexity {mode} \t: {perplexity[mode]}")

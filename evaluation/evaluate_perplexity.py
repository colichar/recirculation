import os
import math
import json
import itertools
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from datetime  import datetime, timezone
from recirculation.dataset import dataloader

MODES = ["vanilla", "recirculated"]


def score_batch_nll(model, window_batch, mode="vanilla"):
    """Score valid next-token targets for a batch of windows.

    Run vanilla or Recirculation inference on the same input IDs and attention
    mask. Logits at each position predict the following token; the first token
    in the window has no within-window predecessor and is not scored. Exclude
    padded targets and compute log probabilities in FP32.

    Args:
        model: RecirculatedGemma instance used for inference.
        window_batch: a tuple of (window_ids, window_mask, document_idx, window_idx)
            entries. The window_ids and window_mask are extracted for nll computation.
            window_ids is a 1-D tensor containing token IDs for one window, initially
            on CPU.
            window_mask is a 1-D tensor containing mask with 1 for present tokens,
            0 for padding.
            The windows must have equal length so they can be stacked.
        mode: "vanilla" or "recirculated" inference.

    Returns:
        A pair (total NLL, scored-token count) across all valid next-token targets
        in the batch.
    """

    # Extract ids, masks and move to model device
    window_ids = torch.stack([window[0] for window in window_batch]).to(model.model.device)
    window_mask = torch.stack([window[1] for window in window_batch]).to(model.model.device)

    if mode=="vanilla":
        logits, targets = model.vanilla_run(window_ids, window_mask)
    else:
        logits, targets = model.recirculated_run(window_ids, window_mask)

    logits = logits[:, :-1, :].float()              # The final logits are predictions for the next token ID
    targets = targets[:, 1:]                        # The token IDs from the next token are targets of the previous final logits
    mask = window_mask[:, 1:]                       # The mask needs to be shifted accordingly

    log_probs = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(
        log_probs.reshape(-1, log_probs.size(-1)),  # Expects [N, C]
        targets.reshape(-1),                        # Expects [N]
        reduction="none"
    )

    valid_nll = nll[mask.reshape(-1).bool()]        # Mask PAD tokens

    return valid_nll.sum().item(), valid_nll.numel()

def evaluate_perplexity(
        model,
        dataloader_config,
        checkpoint = None,
        document_limit=None,
        window_limit=None,
        on_batch=None,
        batch_size = 1
        ):
    """Evaluate vanilla and Recirculation perplexity on paired PG-19 windows.

    Score both modes on the same valid next-token targets, grouping consecutive
    equal-length windows into batches. Each completed batch contributes to the
    cumulative totals and, if a callback is provided, to the checkpoint. On
    resume, continue after the checkpoint's last completed window.

    Args:
        model: RecirculatedGemma instance, including its tokenizer.
        dataloader_config: Keyword arguments for dataloader, excluding tokenizer,
            resume indices, and document_limit.
        checkpoint: Saved last-window cursor and cumulative results, or None for
            a fresh run. Its values initialize the totals; it is not modified.
        document_limit: Cap on documents from the start of the split, or None
            for the full split.
        window_limit: Maximum additional windows to score in this invocation,
            or None to score all remaining selected windows.
        on_batch: Optional callback called after both modes complete a batch.
            Receives the last window's (document_idx, window_idx), followed by
            cumulative NLL sums, token counts, document count, and window count.
        batch_size: Maximum number of independent windows scored together.

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
        start_window_idx = 0
        start_document_idx = 0
        last_scored_document_idx = None
    else:
        total_nll_sum = checkpoint["results"]["total_nll_sum"]
        total_tokens = checkpoint["results"]["total_tokens"]
        total_windows = checkpoint["results"]["total_windows"]
        total_documents = checkpoint["results"]["total_documents"]

        document_idx = checkpoint["last_document_idx"]
        window_idx = checkpoint["last_window_idx"]
        start_document_idx = document_idx
        start_window_idx = window_idx + 1
        last_scored_document_idx = checkpoint["last_document_idx"]

    windows = dataloader(
        tokenizer=model.tokenizer,
        start_document_idx=start_document_idx,
        start_window_idx=start_window_idx,
        document_limit=document_limit,
        **dataloader_config
    )

    if window_limit is not None:
        windows = itertools.islice(windows, window_limit)

    for batch in itertools.batched(
        tqdm(
            windows,
            desc="PG-19 · vanilla + recirc",
            unit=" window",
            dynamic_ncols=True,
        ),
        batch_size):
        nll_sum = {}
        num_tokens = {}

        # Score the batches 
        for mode in MODES:
            nll_sum[mode], num_tokens[mode] = score_batch_nll(
                model=model,
                window_batch=batch,
                mode=mode
            )
        # After all modes are successful register the results
        total_nll_sum = {mode: total_nll_sum[mode] + nll_sum[mode] for mode in MODES}
        total_tokens = {mode: total_tokens[mode] + num_tokens[mode] for mode in MODES}
        total_windows += len(batch)

        # Increment total_documents evaluated
        for _, _, document_idx, window_idx in batch:
            if document_idx != last_scored_document_idx:
                total_documents += 1
                last_scored_document_idx = document_idx

        # Execute callback if provided, e.g. checkpoint saving
        if on_batch is not None:
            on_batch(document_idx, window_idx, total_nll_sum, total_tokens, total_documents, total_windows)


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
    parser.add_argument("--batch-size", type=int, help="Determines the batch size for evaluation")
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
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    batch_size =  args.batch_size if args.batch_size is not None else 1
    if previous_run is not None and batch_size != previous_run.get("batch_size", 1):
        # For example BF16 scores may differ slightly with batch_size
        parser.error("--batch-size must match the saved run")

    # Load model    
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
        "batch_size": batch_size,
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
    if dataloader_config.get("source") == "raw_pg19":
        dataset_name += "-raw-crlf"
    model_name = model_config["model_name"].split("/")[-1]
    # Configs live in inputs/; save alongside that directory regardless of cwd.
    output = args.config.parent.parent / "results" / (
        f"{dataset_name}-{model_name}-{started_at.replace(':', '-')}.json"
    )

    def record_batch(document_idx, window_idx, total_nll_sum, total_tokens, total_documents, total_windows):
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
    perplexity, total_nll_sum, total_tokens, total_documents, total_windows, document_idx, window_idx = evaluate_perplexity(
        model=model,
        dataloader_config=dataloader_config,
        checkpoint=checkpoint,
        document_limit=document_limit,
        window_limit=window_limit,
        on_batch=record_batch,
        batch_size=batch_size
    )

    # Save metadata
    if metadata["checkpoint"] is None:
        metadata["checkpoint"] = checkpoint

    metadata["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()

    save_run_metadata(metadata, output)

    print(f"Evaluation ran for {total_windows} windows and {total_tokens["vanilla"]} tokens\nStopped at document {document_idx} and window {window_idx}\n")
    for mode in MODES:
        print(f"Perplexity {mode} \t: {perplexity[mode]}")

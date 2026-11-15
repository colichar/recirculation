import re
import os
import json
import itertools
from tqdm.auto import tqdm
from recirculation.dataset import dataloader_gsm8k

MODES = ["vanilla", "recirculated"]

def extract_first_number(text: str) -> str:
    """Extract the first number from LLM generated text."""
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)

    if not numbers:
        return ""

    answer = numbers[0]
    if answer.endswith("."):
        answer = answer[:-1]

    return answer

def generate(model, prompt, mode, max_new_tokens=256):
    """Generates text based on a prompt."""
    inputs = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(model.model.device)

    if mode == "vanilla":
        _, generated_ids = model.vanilla_generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
    elif mode == "recirculated":
        _, generated_ids = model.recirculated_generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown generation mode: {mode}")

    generated_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text



def evaluate_gsm8k(
        model,
        dataloader_config,
        checkpoint=None,
        question_limit=None,
        on_batch = None,
        reasoning_limit=256,
        answering_limit=32,
        reasoning_prompt="Q: {question}\nA: Let's think step by step.",
        answering_suffix="Therefore, the answer (arabic numerals) is"
        ):
    """Evaluate the vanilla and recirculated implementation on GSM8k dataset."""
    dataset = dataloader_gsm8k(**dataloader_config)

    if checkpoint is None:
        processed_questions = {mode: 0 for mode in MODES}
        correct_answers = {mode: 0 for mode in MODES}
    else:
        processed_questions = checkpoint["results"]["processed_questions"]
        correct_answers = checkpoint["results"]["correct_answers"]
        dataset = itertools.islice(dataset, processed_questions["vanilla"], None)

    if question_limit is not None:
        dataset = itertools.islice(dataset, question_limit)

    for sample_idx, question, answer in tqdm(dataset, desc="GSM8k", unit=" question", dynamic_ncols=True):
        question_record = {
            "sample_idx": sample_idx,
            "question": question,
            "reference_answer": answer,
            "modes": {},
        }
        for mode in MODES:
            formatted_reasoning_prompt = reasoning_prompt.format(question=question)

            generated_reasoning = generate(
                model=model,
                prompt=formatted_reasoning_prompt,
                mode=mode,
                max_new_tokens=reasoning_limit
            )

            answering_prompt = f"{formatted_reasoning_prompt}\n{generated_reasoning}\n{answering_suffix}"

            generated_answer = generate(
                model=model,
                prompt=answering_prompt,
                mode=mode,
                max_new_tokens=answering_limit
            )

            extracted_answer = extract_first_number(generated_answer)
            is_correct = extracted_answer == answer
            question_record["modes"][mode] = {
                "reasoning": generated_reasoning,
                "answer": generated_answer,
                "extracted_answer": extracted_answer,
                "correct": is_correct,
            }

            if is_correct:
                correct_answers[mode] += 1

            processed_questions[mode] += 1

        if on_batch is not None:
            on_batch(
                processed_questions=processed_questions,
                correct_answers=correct_answers,
                record=question_record,
            )

    if processed_questions["vanilla"] != processed_questions["recirculated"]:
        raise ValueError(f"Scored-token counts differ: {processed_questions}")
    if processed_questions["vanilla"] == 0:
        raise ValueError("No questions were processed.")

    pass_1 = {mode: correct_answers[mode] / processed_questions[mode] for mode in MODES}

    return pass_1, correct_answers, processed_questions

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

def reconcile_generation_log(checkpoint, checkpoint_path):
    """Trim generations beyond the last complete question saved in a checkpoint."""
    if "generation_log" not in checkpoint:
        raise ValueError("Cannot resume: checkpoint has no generation log; start a new logged run.")

    processed_questions = checkpoint["results"]["processed_questions"]
    if processed_questions["vanilla"] != processed_questions["recirculated"]:
        raise ValueError(f"Question counts differ: {processed_questions}")

    log_path = (checkpoint_path.parent / checkpoint["generation_log"]).resolve()
    with log_path.open("r+b") as log:
        for _ in range(processed_questions["vanilla"]):
            line = log.readline()
            if not line.endswith(b"\n"):
                raise ValueError(f"Generation log has fewer complete questions than checkpoint: {log_path}")
            json.loads(line)
        log.truncate()

    return log_path


def save_question_checkpoint(metadata, output, processed_questions, correct_answers, record, log_path=None):
    """Append a completed question's generations before saving its score checkpoint."""
    if processed_questions["vanilla"] != processed_questions["recirculated"]:
        raise ValueError(f"Question counts differ: {processed_questions}")

    if log_path is None:
        log_path = output.with_suffix(".jsonl")
    output.parent.mkdir(exist_ok=True)
    with log_path.open("ab") as log:
        log.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        log.flush()
        os.fsync(log.fileno())

    metadata["checkpoint"] = {
        "generation_log": os.path.relpath(log_path, output.parent),
        "results": {
            "pass@1": {
                mode: correct_answers[mode] / processed_questions[mode]
                for mode in MODES
            },
            "correct_answers": correct_answers.copy(),
            "processed_questions": processed_questions.copy(),
        },
    }
    save_run_metadata(metadata, output)

if __name__ == "__main__":
    import argparse
    import torch
    from pathlib import Path
    from datetime import datetime, timezone
    from recirculation.recirculated_gemma import RecirculatedGemma

    # Load Configs
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="JSON experiment configuration")
    parser.add_argument("--resume-from", type=Path, help="Saved run JSON")
    parser.add_argument("--question-limit", type=int, help="Maximum number of questions to be processed")

    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    # Dataloader Config
    # "dataloader": {
    #   "name": "openai/gsm8k",
    #   "revision": "740312add88f781978c0658806c59bc2815b9866",
    #   "branch": "main"
    #   "split": "test"
    #}
    dataloader_config = config["dataloader"]
    # Evaluation Config
    # "evaluation": {
    #    "mode": "pass@1"
    #    "reasoning_limit": 256,
    #    "answering_limit": 32,
    #    "reasoning_prompt": "Q: {question}\nA: Let's think step by step.",
    #    "answering_suffix": "Therefore, the answer (arabic numerals) is"
    #}
    evaluation_config = config["evaluation"]
    # Model config
    # "model_parameters": {
    #    "model_name": "google/gemma-3-4b-pt",
    #    "model_revision": "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
    #    "source": 16,
    #    "destination": 5,
    #    "alpha": 0.09,
    #    "beta": 1.0,
    #    "alpha_ramp": false,
    #    "dtype": "bfloat16"
    #}
    model_config = config["model_parameters"]

    # Load Checkpoint if provided
    previous_run = None
    if args.resume_from is not None:
        previous_run = json.loads(args.resume_from.read_text(encoding="utf-8"))
        for section in ("dataloader", "model_parameters", "evaluation"):
            if previous_run[section] != config[section]:
                raise ValueError(f"Cannot resume: {section} differs from the saved run.")

    checkpoint = previous_run["checkpoint"] if previous_run is not None else None
    resume_log = reconcile_generation_log(checkpoint, args.resume_from) if checkpoint is not None else None

    # Load model instance
    started_at = datetime.now(timezone.utc).isoformat()

    dtype_by_name = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    model_kwargs = model_config.copy()
    model_kwargs["dtype"] = dtype_by_name[model_kwargs["dtype"]]

    model = RecirculatedGemma(**model_kwargs)

    # Initiate metadata of the run

    import subprocess

    metadata={
        "meta": {
            "evaluation": "gsm8k_pass@1",
            "started_at": started_at,
            "finished_at": None
        },
        "dataloader": dataloader_config,
        "evaluation": evaluation_config,
        "model_parameters": model_config,
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

    # Setup output path for results
    dataset_name = dataloader_config["name"].split("/")[-1]
    evaluation_mode = metadata["evaluation"]["mode"]
    model_name = model_config["model_name"].split("/")[-1]

    output = args.config.parent.parent / "results" / (
        f"{dataset_name}-{evaluation_mode}-{model_name}-{started_at.replace(':', '-')}.json"
    )

    def record_batch(processed_questions, correct_answers, record):
        save_question_checkpoint(metadata, output, processed_questions, correct_answers, record, log_path=resume_log)

    # Run evaluation
    if evaluation_config["mode"] == "pass@1":
        pass_1,  correct_answers, processed_questions = evaluate_gsm8k(
            model=model,
            dataloader_config=dataloader_config,
            checkpoint=checkpoint,
            question_limit=args.question_limit,
            on_batch=record_batch,
            reasoning_limit=evaluation_config["reasoning_limit"],
            answering_limit=evaluation_config["answering_limit"],
            reasoning_prompt=evaluation_config["reasoning_prompt"],
            answering_suffix=evaluation_config["answering_suffix"],
        )
    else:
        raise ValueError("Unknown evaluation mode.")

    if metadata["checkpoint"] is None:
        metadata["checkpoint"] = checkpoint.copy() if checkpoint is not None else None
        if checkpoint is not None:
            metadata["checkpoint"]["generation_log"] = os.path.relpath(resume_log, output.parent)

    metadata["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()

    save_run_metadata(metadata, output)

    print(f"Evaluation processed {processed_questions["vanilla"]} questions\n")
    for mode in MODES:
        print(f"pass@1 {mode} \t: {pass_1[mode]}")

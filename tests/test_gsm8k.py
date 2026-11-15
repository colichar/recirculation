import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from evaluation import evaluate_gsm8k as evaluation


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Therefore, the answer is 18.", "18"),
        ("The result is -3.", "-3"),
        ("First 1,000, then 2", "1000"),
        ("First 1 000, then 2", "1"),
        ("First 1.000, then 2", "1.000"),
        ("There is no number", ""),
    ],
)
def test_extract_first_number(text, expected):
    assert evaluation.extract_first_number(text) == expected


class Inputs(dict):
    def to(self, device):
        assert device == torch.device("cpu")
        return self


class Tokenizer:
    def __init__(self):
        self.calls = []
        self.ids = torch.tensor([[2, 3]])
        self.mask = torch.ones_like(self.ids)

    def __call__(self, prompt, *, return_tensors, add_special_tokens):
        self.calls.append((prompt, return_tensors, add_special_tokens))
        return Inputs(input_ids=self.ids, attention_mask=self.mask)

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        assert ids.tolist() == [7]
        return "Answer: 18"


@pytest.mark.parametrize("mode", evaluation.MODES)
def test_generate_dispatches_to_selected_mode(mode):
    tokenizer = Tokenizer()
    model = SimpleNamespace(
        tokenizer=tokenizer,
        model=SimpleNamespace(device=torch.device("cpu")),
        vanilla_generate=Mock(return_value=(None, torch.tensor([[7]]))),
        recirculated_generate=Mock(return_value=(None, torch.tensor([[7]]))),
    )

    text = evaluation.generate(model, "Q: Test?", mode, max_new_tokens=17)

    assert text == "Answer: 18"
    assert tokenizer.calls == [("Q: Test?", "pt", True)]
    selected = model.vanilla_generate if mode == "vanilla" else model.recirculated_generate
    other = model.recirculated_generate if mode == "vanilla" else model.vanilla_generate
    selected.assert_called_once()
    other.assert_not_called()
    args = selected.call_args.kwargs
    assert args["input_ids"] is tokenizer.ids
    assert args["attention_mask"] is tokenizer.mask
    assert args["max_new_tokens"] == 17


def test_generate_rejects_unknown_mode():
    model = SimpleNamespace(
        tokenizer=Tokenizer(),
        model=SimpleNamespace(device=torch.device("cpu")),
        vanilla_generate=Mock(),
        recirculated_generate=Mock(),
    )

    with pytest.raises(ValueError, match="Unknown generation mode: other"):
        evaluation.generate(model, "Q: Test?", "other")

    model.vanilla_generate.assert_not_called()
    model.recirculated_generate.assert_not_called()


def test_evaluate_gsm8k_keeps_reasoning_and_answers_paired_by_mode(monkeypatch):
    samples = [
        (4, "How many eggs?", "18"),
        (9, "How many bolts?", "3"),
    ]
    monkeypatch.setattr(evaluation, "dataloader_gsm8k", lambda: iter(samples))
    calls = []

    def generate(*, model, prompt, mode, max_new_tokens):
        calls.append((prompt, mode, max_new_tokens))
        question = next(question for _, question, _ in samples if question in prompt)
        if "Therefore, the answer (arabic numerals) is" not in prompt:
            assert prompt == f"Q: {question}\nA: Let's think step by step."
            return f"{mode} reasoning about {question}"

        expected_reasoning = f"{mode} reasoning about {question}"
        other_mode = "recirculated" if mode == "vanilla" else "vanilla"
        assert prompt == (
            f"Q: {question}\nA: Let's think step by step.\n"
            f"{expected_reasoning}\nTherefore, the answer (arabic numerals) is"
        )
        assert f"{other_mode} reasoning" not in prompt
        answers = {
            ("How many eggs?", "vanilla"): "18.",
            ("How many eggs?", "recirculated"): "18.",
            ("How many bolts?", "vanilla"): "no numeric answer",
            ("How many bolts?", "recirculated"): "3 then 999",
        }
        return answers[(question, mode)]

    monkeypatch.setattr(evaluation, "generate", generate)
    records = []

    assert evaluation.evaluate_gsm8k(
        object(), dataloader_config={}, reasoning_limit=123, answering_limit=17,
        on_batch=lambda *, processed_questions, correct_answers, record: records.append(record),
    ) == (
        {"vanilla": 0.5, "recirculated": 1.0},
        {"vanilla": 1, "recirculated": 2},
        {"vanilla": 2, "recirculated": 2},
    )
    assert [(mode, limit) for _, mode, limit in calls] == [
        ("vanilla", 123), ("vanilla", 17),
        ("recirculated", 123), ("recirculated", 17),
    ] * len(samples)
    assert records == [
        {
            "sample_idx": 4,
            "question": "How many eggs?",
            "reference_answer": "18",
            "modes": {
                mode: {
                    "reasoning": f"{mode} reasoning about How many eggs?",
                    "answer": "18.",
                    "extracted_answer": "18",
                    "correct": True,
                }
                for mode in evaluation.MODES
            },
        },
        {
            "sample_idx": 9,
            "question": "How many bolts?",
            "reference_answer": "3",
            "modes": {
                "vanilla": {
                    "reasoning": "vanilla reasoning about How many bolts?",
                    "answer": "no numeric answer",
                    "extracted_answer": "",
                    "correct": False,
                },
                "recirculated": {
                    "reasoning": "recirculated reasoning about How many bolts?",
                    "answer": "3 then 999",
                    "extracted_answer": "3",
                    "correct": True,
                },
            },
        },
    ]


def test_evaluate_gsm8k_resume_matches_uninterrupted_run(monkeypatch):
    samples = [
        (0, "First question?", "1"),
        (1, "Second question?", "2"),
        (2, "Third question?", "3"),
    ]
    monkeypatch.setattr(evaluation, "dataloader_gsm8k", lambda **kwargs: iter(samples))
    calls = []
    answers = {
        (0, "vanilla"): "1", (0, "recirculated"): "wrong",
        (1, "vanilla"): "wrong", (1, "recirculated"): "2",
        (2, "vanilla"): "3", (2, "recirculated"): "wrong",
    }

    def generate(*, model, prompt, mode, max_new_tokens):
        index = next(i for i, question, _ in samples if question in prompt)
        stage = "answer" if "Therefore, the answer (arabic numerals) is" in prompt else "reasoning"
        calls.append((index, mode, stage))
        return answers[(index, mode)] if stage == "answer" else f"{mode} reasoning"

    monkeypatch.setattr(evaluation, "generate", generate)
    checkpoints = []

    def save_checkpoint(*, processed_questions, correct_answers, record):
        checkpoints.append({
            "results": {
                "processed_questions": processed_questions.copy(),
                "correct_answers": correct_answers.copy(),
            }
        })

    first = evaluation.evaluate_gsm8k(
        object(), dataloader_config={}, question_limit=1, on_batch=save_checkpoint
    )
    assert first == (
        {"vanilla": 1.0, "recirculated": 0.0},
        {"vanilla": 1, "recirculated": 0},
        {"vanilla": 1, "recirculated": 1},
    )
    assert checkpoints == [{
        "results": {
            "processed_questions": {"vanilla": 1, "recirculated": 1},
            "correct_answers": {"vanilla": 1, "recirculated": 0},
        }
    }]

    calls.clear()
    # A restarted evaluator reads a fresh checkpoint object from the saved JSON.
    reloaded_checkpoint = json.loads(json.dumps(checkpoints[-1]))
    resumed = evaluation.evaluate_gsm8k(
        object(), dataloader_config={}, checkpoint=reloaded_checkpoint,
        question_limit=2, on_batch=save_checkpoint,
    )
    assert [index for index, _, stage in calls if stage == "reasoning"] == [1, 1, 2, 2]
    assert [checkpoint["results"]["processed_questions"] for checkpoint in checkpoints] == [
        {"vanilla": 1, "recirculated": 1},
        {"vanilla": 2, "recirculated": 2},
        {"vanilla": 3, "recirculated": 3},
    ]

    uninterrupted = evaluation.evaluate_gsm8k(
        object(), dataloader_config={}, question_limit=3
    )
    assert resumed == uninterrupted == (
        {"vanilla": 2 / 3, "recirculated": 1 / 3},
        {"vanilla": 2, "recirculated": 1},
        {"vanilla": 3, "recirculated": 3},
    )


def test_save_question_checkpoint_appends_log_before_checkpoint(tmp_path, monkeypatch):
    output = tmp_path / "results" / "run.json"
    log_path = output.with_suffix(".jsonl")
    metadata = {"checkpoint": None}
    real_save = evaluation.save_run_metadata
    saved_log_lengths = []

    def save_metadata(metadata, output):
        saved_log_lengths.append(len(log_path.read_bytes().splitlines()))
        real_save(metadata, output)

    monkeypatch.setattr(evaluation, "save_run_metadata", save_metadata)
    records = [
        {"sample_idx": 0, "question": "One?", "modes": {"vanilla": {"reasoning": "α"}}},
        {"sample_idx": 1, "question": "Two?", "modes": {"recirculated": {"answer": "2"}}},
    ]

    for count, record in enumerate(records, start=1):
        evaluation.save_question_checkpoint(
            metadata, output,
            {"vanilla": count, "recirculated": count},
            {"vanilla": count, "recirculated": 0},
            record,
        )

    assert saved_log_lengths == [1, 2]
    assert [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] == records
    checkpoint = json.loads(output.read_text(encoding="utf-8"))["checkpoint"]
    assert checkpoint == {
        "generation_log": "run.jsonl",
        "results": {
            "pass@1": {"vanilla": 1.0, "recirculated": 0.0},
            "correct_answers": {"vanilla": 2, "recirculated": 0},
            "processed_questions": {"vanilla": 2, "recirculated": 2},
        },
    }


def test_resume_trims_uncommitted_line_and_reuses_log(tmp_path, monkeypatch):
    first_output = tmp_path / "first" / "run.json"
    metadata = {"checkpoint": None}
    counts = {"vanilla": 1, "recirculated": 1}
    correct = {"vanilla": 1, "recirculated": 0}
    first_record = {"sample_idx": 0, "modes": {"vanilla": {}, "recirculated": {}}}
    second_record = {"sample_idx": 1, "modes": {"vanilla": {}, "recirculated": {}}}
    evaluation.save_question_checkpoint(metadata, first_output, counts, correct, first_record)

    # Simulate an interruption after writing question 1 but before saving its checkpoint.
    def interrupt(*args):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "save_run_metadata", interrupt)
        with pytest.raises(KeyboardInterrupt):
            evaluation.save_question_checkpoint(
                metadata, first_output, {"vanilla": 2, "recirculated": 2},
                correct, second_record,
            )

    saved = json.loads(first_output.read_text(encoding="utf-8"))["checkpoint"]
    log_path = evaluation.reconcile_generation_log(saved, first_output)
    assert [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] == [first_record]

    resumed_output = tmp_path / "resumed" / "run.json"
    evaluation.save_question_checkpoint(
        {"checkpoint": saved}, resumed_output,
        {"vanilla": 2, "recirculated": 2}, correct, second_record, log_path=log_path,
    )
    resumed = json.loads(resumed_output.read_text(encoding="utf-8"))["checkpoint"]
    assert [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] == [
        first_record, second_record,
    ]
    assert evaluation.reconcile_generation_log(resumed, resumed_output) == log_path
    assert resumed["results"]["processed_questions"] == {"vanilla": 2, "recirculated": 2}


def test_resume_rejects_missing_or_short_generation_log(tmp_path):
    checkpoint = {
        "generation_log": "run.jsonl",
        "results": {"processed_questions": {"vanilla": 2, "recirculated": 2}},
    }
    result_path = tmp_path / "run.json"
    (tmp_path / "run.jsonl").write_text('{"sample_idx": 0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="fewer complete questions"):
        evaluation.reconcile_generation_log(checkpoint, result_path)

    with pytest.raises(ValueError, match="checkpoint has no generation log"):
        evaluation.reconcile_generation_log({"results": checkpoint["results"]}, result_path)


def test_resume_discards_partial_trailing_line(tmp_path):
    result_path = tmp_path / "run.json"
    log_path = tmp_path / "run.jsonl"
    log_path.write_bytes(b'{"sample_idx": 0}\n{"sample_idx":')
    checkpoint = {
        "generation_log": log_path.name,
        "results": {"processed_questions": {"vanilla": 1, "recirculated": 1}},
    }

    assert evaluation.reconcile_generation_log(checkpoint, result_path) == log_path
    assert log_path.read_bytes() == b'{"sample_idx": 0}\n'


def test_logged_evaluation_resumes_after_interrupted_checkpoint(tmp_path, monkeypatch):
    samples = [(0, "First?", "1"), (1, "Second?", "2"), (2, "Third?", "3")]
    monkeypatch.setattr(evaluation, "dataloader_gsm8k", lambda **kwargs: iter(samples))
    answers = {
        (0, "vanilla"): "1", (0, "recirculated"): "wrong",
        (1, "vanilla"): "wrong", (1, "recirculated"): "2",
        (2, "vanilla"): "3", (2, "recirculated"): "3",
    }

    def generate(*, model, prompt, mode, max_new_tokens):
        index = next(i for i, question, _ in samples if question in prompt)
        if "Therefore, the answer (arabic numerals) is" in prompt:
            assert f"{mode} reasoning for {index}" in prompt
            return answers[(index, mode)]
        return f"{mode} reasoning for {index}"

    monkeypatch.setattr(evaluation, "generate", generate)
    first_output = tmp_path / "first.json"
    metadata = {"checkpoint": None}
    real_save = evaluation.save_run_metadata

    def interrupt_after_second_append(metadata, output):
        if metadata["checkpoint"]["results"]["processed_questions"]["vanilla"] == 2:
            raise KeyboardInterrupt
        real_save(metadata, output)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "save_run_metadata", interrupt_after_second_append)
        with pytest.raises(KeyboardInterrupt):
            evaluation.evaluate_gsm8k(
                object(), dataloader_config={},
                on_batch=lambda **kwargs: evaluation.save_question_checkpoint(metadata, first_output, **kwargs),
            )

    saved = json.loads(first_output.read_text(encoding="utf-8"))["checkpoint"]
    assert saved["results"]["processed_questions"] == {"vanilla": 1, "recirculated": 1}
    log_path = evaluation.reconcile_generation_log(saved, first_output)
    resumed_output = tmp_path / "resumed.json"
    final = evaluation.evaluate_gsm8k(
        object(), dataloader_config={}, checkpoint=saved,
        on_batch=lambda **kwargs: evaluation.save_question_checkpoint(
            {"checkpoint": None}, resumed_output, log_path=log_path, **kwargs
        ),
    )

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["sample_idx"] for record in records] == [0, 1, 2]
    assert [record["modes"]["vanilla"]["answer"] for record in records] == ["1", "wrong", "3"]
    assert [record["modes"]["recirculated"]["answer"] for record in records] == ["wrong", "2", "3"]
    assert final == (
        {"vanilla": 2 / 3, "recirculated": 2 / 3},
        {"vanilla": 2, "recirculated": 2},
        {"vanilla": 3, "recirculated": 3},
    )
    assert evaluation.reconcile_generation_log(
        json.loads(resumed_output.read_text(encoding="utf-8"))["checkpoint"], resumed_output
    ) == log_path

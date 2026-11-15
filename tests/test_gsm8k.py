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

    assert evaluation.evaluate_gsm8k(object(), reasoning_limit=123, answering_limit=17) == {
        "vanilla": 0.5,
        "recirculated": 1.0,
    }
    assert [(mode, limit) for _, mode, limit in calls] == [
        ("vanilla", 123), ("vanilla", 17),
        ("recirculated", 123), ("recirculated", 17),
    ] * len(samples)

import math
from types import SimpleNamespace

import pytest
import torch

from evaluation import evaluate_perplexity as evaluation_module


class FakeModel:
    def __init__(self, vanilla_logits, recirculated_logits=None):
        self.model = SimpleNamespace(device=torch.device("cpu"))
        self.tokenizer = object()
        self.vanilla_logits = vanilla_logits
        self.recirculated_logits = recirculated_logits or vanilla_logits
        self.calls = []

    def vanilla_run(self, input_ids, attention_mask):
        self.calls.append(("vanilla", input_ids.clone(), attention_mask.clone()))
        return self.vanilla_logits[tuple(input_ids[0].tolist())].unsqueeze(0), input_ids

    def recirculated_run(self, input_ids, attention_mask):
        self.calls.append(("recirculation", input_ids.clone(), attention_mask.clone()))
        return self.recirculated_logits[tuple(input_ids[0].tolist())].unsqueeze(0), input_ids


def test_window_nll_shifts_targets_and_excludes_padded_positions_in_both_modes():
    ids = torch.tensor([2, 1, 2, 0])
    mask = torch.tensor([1, 1, 1, 0])
    vanilla = torch.tensor([
        [0, 3, 0],   # predicts the first text token (1)
        [0, 0, 3],   # predicts the second text token (2)
        [30, 0, 0],  # predicts PAD; must not contribute
        [0, 30, 0],  # final logits have no target
    ], dtype=torch.bfloat16)
    recirculated = vanilla.clone()
    recirculated[1] = torch.tensor([3, 0, 0])
    model = FakeModel({tuple(ids.tolist()): vanilla}, {tuple(ids.tolist()): recirculated})

    vanilla_sum, vanilla_count = evaluation_module.score_window_nll(model, ids, mask, mode="vanilla")
    recirc_sum, recirc_count = evaluation_module.score_window_nll(model, ids, mask, mode="recirculated")

    correct_loss = math.log1p(2 * math.exp(-3))
    assert vanilla_count == recirc_count == 2
    assert vanilla_sum == pytest.approx(2 * correct_loss, abs=1e-6)
    assert recirc_sum == pytest.approx(correct_loss + 3 + correct_loss, abs=1e-6)
    assert math.isfinite(vanilla_sum) and math.isfinite(recirc_sum)
    assert [name for name, _, _ in model.calls] == ["vanilla", "recirculation"]
    for _, seen_ids, seen_mask in model.calls:
        torch.testing.assert_close(seen_ids, ids.unsqueeze(0))
        torch.testing.assert_close(seen_mask, mask.unsqueeze(0))


def test_perplexity_weights_windows_by_valid_token_count(monkeypatch):
    full_ids = torch.tensor([0, 1, 1])
    partial_ids = torch.tensor([0, 0, 0])
    windows = [
        (full_ids, torch.tensor([1, 1, 1])),
        (partial_ids, torch.tensor([1, 1, 0])),
    ]
    logits = {
        tuple(full_ids.tolist()): torch.zeros(3, 2),   # two targets, NLL = 2 * ln(2)
        tuple(partial_ids.tolist()): torch.tensor([
            [0, math.log(3)],  # valid target 0 has probability 1/4
            [0, 0],            # PAD target is excluded
            [0, 0],
        ]),
    }
    recirculated_logits = {key: value.clone() for key, value in logits.items()}
    recirculated_logits[tuple(partial_ids.tolist())][0] = torch.tensor([math.log(3), 0])
    model = FakeModel(logits, recirculated_logits)

    def dataloader(*, tokenizer, document_limit):
        assert tokenizer is model.tokenizer
        assert document_limit == 2
        yield from windows

    monkeypatch.setattr(evaluation_module, "dataloader", dataloader)

    perplexity, counts, window_count = evaluation_module.evaluate_perplexity(model, document_limit=2)

    # Three valid targets total; averaging the two window perplexities gives 3, not this value.
    vanilla_expected = math.exp((2 * math.log(2) + math.log(4)) / 3)
    recirculated_expected = math.exp((2 * math.log(2) + math.log(4 / 3)) / 3)
    assert window_count == 2
    assert counts == {"vanilla": 3, "recirculated": 3}
    assert perplexity["vanilla"] == pytest.approx(vanilla_expected, rel=1e-6)
    assert perplexity["recirculated"] == pytest.approx(recirculated_expected, rel=1e-6)
    assert [name for name, _, _ in model.calls] == [
        "vanilla", "recirculation", "vanilla", "recirculation"
    ]
    for index, (ids, mask) in enumerate(windows):
        for _, seen_ids, seen_mask in model.calls[2 * index : 2 * index + 2]:
            torch.testing.assert_close(seen_ids, ids.unsqueeze(0))
            torch.testing.assert_close(seen_mask, mask.unsqueeze(0))

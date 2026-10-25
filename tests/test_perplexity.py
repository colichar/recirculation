import math
import json
from types import SimpleNamespace

import pytest
import torch

from evaluation import evaluate_perplexity as evaluation_module
from recirculation import dataset as dataset_module
from test_dataset import Documents, Tokenizer, document


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


def test_window_nll_without_bos_excludes_first_text_token():
    ids = torch.tensor([1, 2, 3])
    mask = torch.ones_like(ids)
    logits = torch.tensor([
        [0, 0, 8, 0],  # predicts text token 2
        [0, 0, 0, 8],  # predicts text token 3
        [0, 8, 0, 0],  # would predict token 1 if final logits were scored
    ], dtype=torch.float32)
    model = FakeModel({tuple(ids.tolist()): logits})

    nll_sum, count = evaluation_module.score_window_nll(model, ids, mask)

    assert count == 2
    assert nll_sum == pytest.approx(2 * math.log1p(3 * math.exp(-8)), abs=1e-6)


def test_perplexity_weights_windows_by_valid_token_count(monkeypatch):
    full_ids = torch.tensor([0, 1, 1])
    partial_ids = torch.tensor([0, 0, 0])
    windows = [
        (full_ids, torch.tensor([1, 1, 1]), 0, 0),
        (partial_ids, torch.tensor([1, 1, 0]), 1, 0),
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

    def dataloader(*, tokenizer, document_limit, start_document_idx, start_window_idx):
        assert tokenizer is model.tokenizer
        assert document_limit == 2
        assert (start_document_idx, start_window_idx) == (0, 0)
        yield from windows

    monkeypatch.setattr(evaluation_module, "dataloader", dataloader)

    perplexity, nll_sums, counts, document_count, window_count, document_idx, window_idx = (
        evaluation_module.evaluate_perplexity(model, dataloader_config={}, document_limit=2)
    )

    # Three valid targets total; averaging the two window perplexities gives 3, not this value.
    vanilla_expected = math.exp((2 * math.log(2) + math.log(4)) / 3)
    recirculated_expected = math.exp((2 * math.log(2) + math.log(4 / 3)) / 3)
    assert window_count == 2
    assert document_count == 2
    assert (document_idx, window_idx) == (1, 0)
    assert counts == {"vanilla": 3, "recirculated": 3}
    assert nll_sums["vanilla"] == pytest.approx(2 * math.log(2) + math.log(4), rel=1e-6)
    assert perplexity["vanilla"] == pytest.approx(vanilla_expected, rel=1e-6)
    assert perplexity["recirculated"] == pytest.approx(recirculated_expected, rel=1e-6)
    assert [name for name, _, _ in model.calls] == [
        "vanilla", "recirculation", "vanilla", "recirculation"
    ]
    for index, (ids, mask, _, _) in enumerate(windows):
        for _, seen_ids, seen_mask in model.calls[2 * index : 2 * index + 2]:
            torch.testing.assert_close(seen_ids, ids.unsqueeze(0))
            torch.testing.assert_close(seen_mask, mask.unsqueeze(0))


def test_checkpoint_after_interruption_resumes_across_documents(monkeypatch, tmp_path):
    documents = [document([1, 2, 3, 4]), document([5, 6])]
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: Documents(documents))
    model = SimpleNamespace(tokenizer=Tokenizer())
    seen = []

    def score_window_nll(model, window_ids, window_mask, mode):
        seen.append((tuple(window_ids.tolist()), mode))
        return float(window_ids[1]) + (0.5 if mode == "recirculated" else 0), int(window_mask[1:].sum())

    monkeypatch.setattr(evaluation_module, "score_window_nll", score_window_nll)
    output = tmp_path / "checkpoint.json"

    def save_window(document_idx, window_idx, nll_sums, counts, documents_count, windows_count):
        evaluation_module.save_run_metadata({"checkpoint": {
            "last_document_idx": document_idx,
            "last_window_idx": window_idx,
            "results": {
                "total_nll_sum": nll_sums.copy(),
                "total_tokens": counts.copy(),
                "total_documents": documents_count,
                "total_windows": windows_count,
            },
        }}, output)
        if windows_count == 2:
            raise RuntimeError("interrupted after checkpoint")

    with pytest.raises(RuntimeError, match="interrupted after checkpoint"):
        evaluation_module.evaluate_perplexity(
            model, {"window_size": 2}, on_window=save_window
        )

    assert not output.with_name(output.name + ".tmp").exists()
    checkpoint = json.loads(output.read_text())["checkpoint"]
    assert (checkpoint["last_document_idx"], checkpoint["last_window_idx"]) == (0, 1)
    assert (checkpoint["results"]["total_documents"], checkpoint["results"]["total_windows"]) == (1, 2)
    assert checkpoint["results"]["total_tokens"] == {"vanilla": 4, "recirculated": 4}

    resumed_windows = []
    resumed = evaluation_module.evaluate_perplexity(
        model, {"window_size": 2}, checkpoint=checkpoint, on_window=lambda *args: resumed_windows.append(args[:2])
    )
    assert resumed_windows == [(1, 0)]
    uninterrupted = evaluation_module.evaluate_perplexity(model, {"window_size": 2})
    assert resumed == uninterrupted
    assert (resumed[3], resumed[4], resumed[5:]) == (2, 3, (1, 0))
    assert len(seen) == 12  # 2 modes for 2 + 1 + 3 windows

    no_more_windows = []
    final_checkpoint = {
        "last_document_idx": resumed[5],
        "last_window_idx": resumed[6],
        "results": {
            "total_nll_sum": resumed[1],
            "total_tokens": resumed[2],
            "total_documents": resumed[3],
            "total_windows": resumed[4],
        },
    }
    assert evaluation_module.evaluate_perplexity(
        model, {"window_size": 2}, checkpoint=final_checkpoint,
        on_window=lambda *args: no_more_windows.append(args)
    ) == uninterrupted
    assert no_more_windows == []


def test_failed_second_mode_keeps_last_complete_window_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        dataset_module, "load_dataset",
        lambda *args, **kwargs: Documents([document([1, 2, 3, 4])]),
    )
    model = SimpleNamespace(tokenizer=Tokenizer())
    fail_on_second_window = True

    def score_window_nll(model, window_ids, window_mask, mode):
        if fail_on_second_window and window_ids[1].item() == 3 and mode == "recirculated":
            raise RuntimeError("recirculation failed")
        return float(window_ids[1]) + (0.5 if mode == "recirculated" else 0), int(window_mask[1:].sum())

    monkeypatch.setattr(evaluation_module, "score_window_nll", score_window_nll)
    output = tmp_path / "checkpoint.json"
    saved_windows = []

    def save_window(document_idx, window_idx, nll_sums, counts, documents_count, windows_count):
        saved_windows.append((document_idx, window_idx))
        evaluation_module.save_run_metadata({"checkpoint": {
            "last_document_idx": document_idx,
            "last_window_idx": window_idx,
            "results": {
                "total_nll_sum": nll_sums.copy(),
                "total_tokens": counts.copy(),
                "total_documents": documents_count,
                "total_windows": windows_count,
            },
        }}, output)

    with pytest.raises(RuntimeError, match="recirculation failed"):
        evaluation_module.evaluate_perplexity(model, {"window_size": 2}, on_window=save_window)

    checkpoint = json.loads(output.read_text())["checkpoint"]
    assert saved_windows == [(0, 0)]
    assert (checkpoint["last_document_idx"], checkpoint["last_window_idx"]) == (0, 0)
    assert checkpoint["results"]["total_nll_sum"] == {"vanilla": 1.0, "recirculated": 1.5}
    assert checkpoint["results"]["total_tokens"] == {"vanilla": 2, "recirculated": 2}
    assert checkpoint["results"]["total_windows"] == checkpoint["results"]["total_documents"] == 1

    fail_on_second_window = False
    resumed = evaluation_module.evaluate_perplexity(model, {"window_size": 2}, checkpoint=checkpoint)
    uninterrupted = evaluation_module.evaluate_perplexity(model, {"window_size": 2})
    assert resumed == uninterrupted
    assert (resumed[5], resumed[6]) == (0, 1)


def test_first_only_resume_within_and_across_documents_matches_full_run(monkeypatch):
    documents = [document(range(100, 100 + 40 + 41 + 32)), document(range(300, 340))]
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: Documents(documents))
    model = SimpleNamespace(tokenizer=Tokenizer())
    config = {"window_size": 40, "window_prefix": "first_only"}

    def score_window_nll(model, window_ids, window_mask, mode):
        return float(window_ids[1]) + (0.5 if mode == "recirculated" else 0), int(window_mask[1:].sum())

    monkeypatch.setattr(evaluation_module, "score_window_nll", score_window_nll)

    def checkpoint(result):
        _, nll_sums, counts, document_count, window_count, document_idx, window_idx = result
        return {
            "last_document_idx": document_idx,
            "last_window_idx": window_idx,
            "results": {
                "total_nll_sum": nll_sums.copy(),
                "total_tokens": counts.copy(),
                "total_documents": document_count,
                "total_windows": window_count,
            },
        }

    first = evaluation_module.evaluate_perplexity(model, config, window_limit=1)
    second = evaluation_module.evaluate_perplexity(model, config, checkpoint=checkpoint(first), window_limit=1)
    completed = evaluation_module.evaluate_perplexity(model, config, checkpoint=checkpoint(second))
    full = evaluation_module.evaluate_perplexity(model, config)

    assert (first[3], first[4], first[5:]) == (1, 1, (0, 0))
    assert (second[3], second[4], second[5:]) == (1, 2, (0, 1))
    assert completed == full
    assert completed[2] == {"vanilla": 120, "recirculated": 120}
    assert (completed[3], completed[4], completed[5:]) == (2, 3, (1, 0))

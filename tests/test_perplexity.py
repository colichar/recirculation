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
        return torch.stack([self.vanilla_logits[tuple(row.tolist())] for row in input_ids]), input_ids

    def recirculated_run(self, input_ids, attention_mask):
        self.calls.append(("recirculation", input_ids.clone(), attention_mask.clone()))
        return torch.stack([self.recirculated_logits[tuple(row.tolist())] for row in input_ids]), input_ids


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

    batch = ((ids, mask, 0, 0),)
    vanilla_sum, vanilla_count = evaluation_module.score_batch_nll(model, batch, mode="vanilla")
    recirc_sum, recirc_count = evaluation_module.score_batch_nll(model, batch, mode="recirculated")

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

    nll_sum, count = evaluation_module.score_batch_nll(model, ((ids, mask, 0, 0),))

    assert count == 2
    assert nll_sum == pytest.approx(2 * math.log1p(3 * math.exp(-8)), abs=1e-6)


def test_batch_scores_same_targets_as_independent_windows():
    ids_a = torch.tensor([0, 1, 2, 1])
    ids_b = torch.tensor([2, 2, 1, 0])
    mask = torch.ones(4, dtype=torch.long)
    windows = ((ids_a, mask, 0, 0), (ids_b, mask, 1, 0))
    logits = {
        tuple(ids_a.tolist()): torch.tensor([[0, 3, 0], [0, 0, 3], [0, 3, 0], [10, 0, 0]], dtype=torch.bfloat16),
        tuple(ids_b.tolist()): torch.tensor([[0, 0, 2], [0, 2, 0], [2, 0, 0], [0, 10, 0]], dtype=torch.bfloat16),
    }
    recirculated = {key: value.clone() for key, value in logits.items()}
    recirculated[tuple(ids_b.tolist())][1] = torch.tensor([2, 0, 0])
    model = FakeModel(logits, recirculated)

    for mode, mode_logits in (("vanilla", logits), ("recirculated", recirculated)):
        single_scores = [evaluation_module.score_batch_nll(model, (window,), mode) for window in windows]
        batch_sum, batch_count = evaluation_module.score_batch_nll(model, windows, mode)
        target_nll = [
            -torch.log_softmax(mode_logits[tuple(ids.tolist())][:-1].float(), dim=-1)
            .gather(1, ids[1:, None]).flatten()
            for ids, _, _, _ in windows
        ]
        assert batch_count == sum(count for _, count in single_scores) == 6
        assert batch_sum == pytest.approx(sum(score for score, _ in single_scores), abs=1e-6)
        assert batch_sum == pytest.approx(torch.cat(target_nll).sum().item(), abs=1e-6)

    for mode, seen_ids, seen_mask in model.calls:
        torch.testing.assert_close(seen_mask, torch.ones_like(seen_ids))
        if seen_ids.shape[0] == 2:
            torch.testing.assert_close(seen_ids, torch.stack([ids_a, ids_b]))


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


def test_batches_cross_documents_and_window_limit_keeps_partial_batch(monkeypatch):
    # Document 2 yields no windows; it must not count as a scored document.
    windows = [
        (torch.tensor([0, 1, 2, 1]), torch.ones(4, dtype=torch.long), doc, idx)
        for doc, idx in ((0, 0), (0, 1), (1, 0), (3, 0), (3, 1))
    ]
    # Distinguish windows in the same document so an incorrect shared row is visible.
    windows[1] = (torch.tensor([0, 2, 1, 2]), windows[1][1], 0, 1)
    windows[2] = (torch.tensor([1, 2, 0, 1]), windows[2][1], 1, 0)
    windows[3] = (torch.tensor([2, 0, 1, 2]), windows[3][1], 3, 0)
    windows[4] = (torch.tensor([2, 1, 0, 2]), windows[4][1], 3, 1)
    logits = {
        tuple(ids.tolist()): torch.tensor([[0., 1., 2.], [2., 0., 1.], [1., 2., 0.], [0., 0., 0.]]) + i * 0.1
        for i, (ids, _, _, _) in enumerate(windows)
    }
    recirculated = {key: value + torch.tensor([0.2, 0., -0.2]) for key, value in logits.items()}
    model = FakeModel(logits, recirculated)

    def dataloader(*, tokenizer, start_document_idx, start_window_idx, document_limit):
        assert tokenizer is model.tokenizer
        for window in windows:
            if window[2] >= start_document_idx and window[2] < document_limit:
                if window[2] != start_document_idx or window[3] >= start_window_idx:
                    yield window

    monkeypatch.setattr(evaluation_module, "dataloader", dataloader)

    reference = evaluation_module.evaluate_perplexity(model, {}, document_limit=4, batch_size=1)
    completed_batches = []
    batched = evaluation_module.evaluate_perplexity(
        model, {}, document_limit=4, batch_size=3,
        on_batch=lambda *args: completed_batches.append((args[0], args[1], args[4], args[5])),
    )
    assert batched[2:] == reference[2:] == ({"vanilla": 15, "recirculated": 15}, 3, 5, 3, 1)
    for mode in evaluation_module.MODES:
        assert batched[1][mode] == pytest.approx(reference[1][mode], abs=1e-5)
    assert completed_batches == [(1, 0, 2, 3), (3, 1, 3, 5)]

    limited_batches = []
    limited = evaluation_module.evaluate_perplexity(
        model, {}, document_limit=4, batch_size=3, window_limit=4,
        on_batch=lambda *args: limited_batches.append((args[0], args[1], args[5])),
    )
    assert limited_batches == [(1, 0, 3), (3, 0, 4)]
    assert (limited[2], limited[3], limited[4], limited[5:]) == (
        {"vanilla": 12, "recirculated": 12}, 3, 4, (3, 0),
    )
    checkpoint = {
        "last_document_idx": limited[5],
        "last_window_idx": limited[6],
        "results": {
            "total_nll_sum": limited[1].copy(),
            "total_tokens": limited[2].copy(),
            "total_documents": limited[3],
            "total_windows": limited[4],
        },
    }
    resumed = evaluation_module.evaluate_perplexity(model, {}, document_limit=4, batch_size=3, checkpoint=checkpoint)
    assert resumed[2:] == batched[2:]
    for mode in evaluation_module.MODES:
        assert resumed[1][mode] == pytest.approx(batched[1][mode], abs=1e-5)


def test_checkpoint_after_interruption_resumes_across_documents(monkeypatch, tmp_path):
    documents = [document([1, 2, 3, 4]), document([5, 6])]
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: Documents(documents))
    model = SimpleNamespace(tokenizer=Tokenizer())
    seen = []

    def score_batch_nll(model, window_batch, mode):
        seen.append((tuple(tuple(ids.tolist()) for ids, _, _, _ in window_batch), mode))
        return (sum(float(ids[1]) + (0.5 if mode == "recirculated" else 0) for ids, _, _, _ in window_batch),
                sum(int(mask[1:].sum()) for _, mask, _, _ in window_batch))

    monkeypatch.setattr(evaluation_module, "score_batch_nll", score_batch_nll)
    output = tmp_path / "checkpoint.json"

    def save_batch(document_idx, window_idx, nll_sums, counts, documents_count, windows_count):
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
            model, {"window_size": 2}, batch_size=2, on_batch=save_batch
        )

    assert not output.with_name(output.name + ".tmp").exists()
    checkpoint = json.loads(output.read_text())["checkpoint"]
    assert (checkpoint["last_document_idx"], checkpoint["last_window_idx"]) == (0, 1)
    assert (checkpoint["results"]["total_documents"], checkpoint["results"]["total_windows"]) == (1, 2)
    assert checkpoint["results"]["total_tokens"] == {"vanilla": 4, "recirculated": 4}

    resumed_batches = []
    resumed = evaluation_module.evaluate_perplexity(
        model, {"window_size": 2}, batch_size=2, checkpoint=checkpoint,
        on_batch=lambda *args: resumed_batches.append(args[:2]),
    )
    assert resumed_batches == [(1, 0)]
    uninterrupted = evaluation_module.evaluate_perplexity(model, {"window_size": 2}, batch_size=2)
    assert resumed == uninterrupted
    assert (resumed[3], resumed[4], resumed[5:]) == (2, 3, (1, 0))
    assert len(seen) == 8  # 2 modes for 1 + 1 + 2 batches

    no_more_batches = []
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
        model, {"window_size": 2}, batch_size=2, checkpoint=final_checkpoint,
        on_batch=lambda *args: no_more_batches.append(args)
    ) == uninterrupted
    assert no_more_batches == []


def test_failed_second_mode_keeps_last_complete_batch_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        dataset_module, "load_dataset",
        lambda *args, **kwargs: Documents([document(range(1, 9)), document([9, 10])]),
    )
    model = SimpleNamespace(tokenizer=Tokenizer())
    fail_on_second_batch = True

    def score_batch_nll(model, window_batch, mode):
        if fail_on_second_batch and mode == "recirculated" and any(ids[1].item() == 5 for ids, _, _, _ in window_batch):
            raise RuntimeError("recirculation failed")
        return (sum(float(ids[1]) + (0.5 if mode == "recirculated" else 0) for ids, _, _, _ in window_batch),
                sum(int(mask[1:].sum()) for _, mask, _, _ in window_batch))

    monkeypatch.setattr(evaluation_module, "score_batch_nll", score_batch_nll)
    output = tmp_path / "checkpoint.json"
    saved_batches = []

    def save_batch(document_idx, window_idx, nll_sums, counts, documents_count, windows_count):
        saved_batches.append((document_idx, window_idx))
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
        evaluation_module.evaluate_perplexity(model, {"window_size": 2}, batch_size=2, on_batch=save_batch)

    checkpoint = json.loads(output.read_text())["checkpoint"]
    assert saved_batches == [(0, 1)]
    assert (checkpoint["last_document_idx"], checkpoint["last_window_idx"]) == (0, 1)
    assert checkpoint["results"]["total_nll_sum"] == {"vanilla": 4.0, "recirculated": 5.0}
    assert checkpoint["results"]["total_tokens"] == {"vanilla": 4, "recirculated": 4}
    assert (checkpoint["results"]["total_windows"], checkpoint["results"]["total_documents"]) == (2, 1)

    fail_on_second_batch = False
    resumed = evaluation_module.evaluate_perplexity(model, {"window_size": 2}, batch_size=2, checkpoint=checkpoint)
    uninterrupted = evaluation_module.evaluate_perplexity(model, {"window_size": 2}, batch_size=2)
    assert resumed == uninterrupted
    assert (resumed[3], resumed[4], resumed[5:]) == (2, 5, (1, 0))


def test_first_only_resume_within_and_across_documents_matches_full_run(monkeypatch):
    documents = [document(range(100, 100 + 40 + 41 + 32)), document(range(300, 340))]
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: Documents(documents))
    model = SimpleNamespace(tokenizer=Tokenizer())
    config = {"window_size": 40, "window_prefix": "first_only"}

    def score_batch_nll(model, window_batch, mode):
        return (sum(float(ids[1]) + (0.5 if mode == "recirculated" else 0) for ids, _, _, _ in window_batch),
                sum(int(mask[1:].sum()) for _, mask, _, _ in window_batch))

    monkeypatch.setattr(evaluation_module, "score_batch_nll", score_batch_nll)

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

    first = evaluation_module.evaluate_perplexity(model, config, batch_size=2, window_limit=1)
    second = evaluation_module.evaluate_perplexity(model, config, batch_size=2, checkpoint=checkpoint(first), window_limit=1)
    completed = evaluation_module.evaluate_perplexity(model, config, batch_size=2, checkpoint=checkpoint(second))
    full = evaluation_module.evaluate_perplexity(model, config, batch_size=2)

    assert (first[3], first[4], first[5:]) == (1, 1, (0, 0))
    assert (second[3], second[4], second[5:]) == (1, 2, (0, 1))
    assert completed == full
    assert completed[2] == {"vanilla": 120, "recirculated": 120}
    assert (completed[3], completed[4], completed[5:]) == (2, 3, (1, 0))

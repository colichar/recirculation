import torch

from recirculation import dataset as dataset_module


class Tokenizer:
    bos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert not add_special_tokens
        ids = torch.tensor([[int(token) for token in text.split()]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class Documents:
    def __init__(self, documents):
        self.documents = documents

    def __iter__(self):
        return iter(self.documents)

    def take(self, count):
        return Documents(self.documents[:count])

    def skip(self, count):
        return Documents(self.documents[count:])


def document(tokens):
    return {"text": " ".join(map(str, tokens))}


def test_windows_keep_text_order_and_reset_at_document_boundaries(monkeypatch):
    first = list(range(100, 100 + 2 * 1023))
    second = list(range(3000, 3000 + 1023))
    monkeypatch.setattr(
        dataset_module, "load_dataset", lambda *args, **kwargs: Documents([document(first), document(second)])
    )

    windows = list(dataset_module.dataloader(Tokenizer()))

    assert len(windows) == 3
    for (ids, mask, document_idx, window_idx), expected_text, expected_cursor in zip(
        windows, (first[:1023], first[1023:], second), ((0, 0), (0, 1), (1, 0)), strict=True
    ):
        assert (document_idx, window_idx) == expected_cursor
        assert ids.device.type == "cpu"
        assert ids.shape == mask.shape == (1024,)
        assert ids.tolist() == [Tokenizer.bos_token_id, *expected_text]
        assert mask.tolist() == [1] * 1024


def test_final_fragment_31_is_discarded_and_32_is_padded(monkeypatch):
    text = list(range(100, 100 + 1023 + 32))
    documents = [document(text[:1023 + 31]), document(text)]
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: Documents(documents))

    first_only = list(dataset_module.dataloader(Tokenizer(), document_limit=1))
    assert len(first_only) == 1

    windows = list(dataset_module.dataloader(Tokenizer()))
    assert len(windows) == 3
    ids, mask, document_idx, window_idx = windows[-1]
    assert (document_idx, window_idx) == (1, 1)
    assert ids.shape == mask.shape == (1024,)
    assert ids[:33].tolist() == [Tokenizer.bos_token_id, *text[1023:]]
    assert ids[33:].tolist() == [Tokenizer.pad_token_id] * (1024 - 33)
    assert mask.tolist() == [1] * 33 + [0] * (1024 - 33)
    assert int(mask[1:].sum()) == 32


def test_resume_after_padded_final_window_advances_to_next_document(monkeypatch):
    first = list(range(100, 100 + 40 + 32))
    second = list(range(200, 200 + 40))
    monkeypatch.setattr(
        dataset_module, "load_dataset",
        lambda *args, **kwargs: Documents([document(first), document(second)]),
    )

    first_document = list(dataset_module.dataloader(Tokenizer(), window_size=40, document_limit=1))
    assert [(doc_idx, win_idx) for _, _, doc_idx, win_idx in first_document] == [(0, 0), (0, 1)]
    assert first_document[-1][1].tolist() == [1] * 33 + [0] * 8

    resumed = list(dataset_module.dataloader(
        Tokenizer(), window_size=40, document_limit=2,
        start_document_idx=0, start_window_idx=2,
    ))
    assert len(resumed) == 1
    ids, mask, document_idx, window_idx = resumed[0]
    assert (document_idx, window_idx) == (1, 0)
    assert ids.tolist() == [Tokenizer.bos_token_id, *second]
    assert mask.tolist() == [1] * 41


def test_first_only_packs_full_windows_and_resumes_across_documents(monkeypatch):
    first = list(range(100, 100 + 40 + 41 + 32))
    second = list(range(300, 300 + 40))
    monkeypatch.setattr(
        dataset_module, "load_dataset",
        lambda *args, **kwargs: Documents([document(first), document(second)]),
    )

    kwargs = {"window_size": 40, "window_prefix": "first_only"}
    windows = list(dataset_module.dataloader(Tokenizer(), **kwargs))
    assert [(doc_idx, win_idx) for _, _, doc_idx, win_idx in windows] == [
        (0, 0), (0, 1), (1, 0)
    ]
    assert [ids.tolist() for ids, _, _, _ in windows] == [
        [Tokenizer.bos_token_id, *first[:40]],
        first[40:81],
        [Tokenizer.bos_token_id, *second],
    ]
    for ids, mask, _, _ in windows:
        assert ids.shape == mask.shape == (41,)
        assert mask.tolist() == [1] * 41
        assert int(mask[1:].sum()) == 40

    within_document = list(dataset_module.dataloader(
        Tokenizer(), **kwargs, start_document_idx=0, start_window_idx=1,
    ))
    across_documents = list(dataset_module.dataloader(
        Tokenizer(), **kwargs, start_document_idx=0, start_window_idx=2,
    ))
    skipped_document = list(dataset_module.dataloader(
        Tokenizer(), **kwargs, start_document_idx=1, start_window_idx=0,
    ))
    assert [(doc_idx, win_idx) for _, _, doc_idx, win_idx in within_document] == [(0, 1), (1, 0)]
    assert [(doc_idx, win_idx) for _, _, doc_idx, win_idx in across_documents] == [(1, 0)]
    assert [(doc_idx, win_idx) for _, _, doc_idx, win_idx in skipped_document] == [(1, 0)]
    for actual, expected in zip(within_document, windows[1:], strict=True):
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(across_documents[0][0], windows[-1][0])
    torch.testing.assert_close(across_documents[0][1], windows[-1][1])
    torch.testing.assert_close(skipped_document[0][0], windows[-1][0])
    torch.testing.assert_close(skipped_document[0][1], windows[-1][1])


def test_document_limit_and_pinned_streaming_split(monkeypatch):
    documents = [document([value] * 1023) for value in (10, 20, 30)]
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return Documents(documents)

    monkeypatch.setattr(dataset_module, "load_dataset", load_dataset)

    limited = list(dataset_module.dataloader(Tokenizer(), document_limit=1))
    full = list(dataset_module.dataloader(Tokenizer(), document_limit=None))

    assert len(limited) == 1
    assert limited[0][0][1].item() == 10
    assert [ids[1].item() for ids, _, _, _ in full] == [10, 20, 30]
    assert calls == [
        (("emozilla/pg19",), {
            "split": "test",
            "streaming": True,
            "revision": "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8",
        })
    ] * 2

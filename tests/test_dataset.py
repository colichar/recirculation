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


def document(tokens):
    return {"text": " ".join(map(str, tokens))}


def test_windows_keep_text_order_and_reset_at_document_boundaries(monkeypatch):
    first = list(range(100, 100 + 2 * 1023))
    second = list(range(3000, 3000 + 1023))
    monkeypatch.setattr(
        dataset_module, "load_dataset", lambda *args, **kwargs: Documents([document(first), document(second)])
    )

    windows = list(dataset_module.preprocess_dataset(Tokenizer()))

    assert len(windows) == 3
    for (ids, mask), expected_text in zip(
        windows, (first[:1023], first[1023:], second), strict=True
    ):
        assert ids.device.type == "cpu"
        assert ids.shape == mask.shape == (1024,)
        assert ids.tolist() == [Tokenizer.bos_token_id, *expected_text]
        assert mask.tolist() == [1] * 1024


def test_final_fragment_31_is_discarded_and_32_is_padded(monkeypatch):
    text = list(range(100, 100 + 1023 + 32))
    documents = [document(text[:1023 + 31]), document(text)]
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: Documents(documents))

    first_only = list(dataset_module.preprocess_dataset(Tokenizer(), document_limit=1))
    assert len(first_only) == 1

    windows = list(dataset_module.preprocess_dataset(Tokenizer()))
    assert len(windows) == 3
    ids, mask = windows[-1]
    assert ids.shape == mask.shape == (1024,)
    assert ids[:33].tolist() == [Tokenizer.bos_token_id, *text[1023:]]
    assert ids[33:].tolist() == [Tokenizer.pad_token_id] * (1024 - 33)
    assert mask.tolist() == [1] * 33 + [0] * (1024 - 33)
    assert int(mask[1:].sum()) == 32


def test_document_limit_and_pinned_streaming_split(monkeypatch):
    documents = [document([value] * 1023) for value in (10, 20, 30)]
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return Documents(documents)

    monkeypatch.setattr(dataset_module, "load_dataset", load_dataset)

    limited = list(dataset_module.preprocess_dataset(Tokenizer(), document_limit=1))
    full = list(dataset_module.preprocess_dataset(Tokenizer(), document_limit=None))

    assert len(limited) == 1
    assert limited[0][0][1].item() == 10
    assert [ids[1].item() for ids, _ in full] == [10, 20, 30]
    assert calls == [
        (("emozilla/pg19",), {
            "split": "test",
            "streaming": True,
            "revision": "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8",
        })
    ] * 2

import importlib
from pathlib import Path

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig


@pytest.fixture
def tiny_gemma(monkeypatch):
    # The research module currently uses a sibling import when run as a script.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "recirculation"))
    module = importlib.import_module("recirculated_gemma")
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        sliding_window=4,
        max_position_embeddings=32,
        layer_types=["sliding_attention", "full_attention"] * 2,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = Gemma3ForCausalLM(config).eval()
    # Exercise the real wrapper initialization without downloading weights or a tokenizer.
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model)
    return module.RecirculatedGemma(source=2, destination=0)


@pytest.mark.parametrize("length", [1, 7])
def test_zero_alpha_matches_vanilla_across_sliding_window(tiny_gemma, length):
    tiny_gemma.alpha = 0.0
    input_ids = torch.arange(2, 2 + length).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    expected, _ = tiny_gemma.run_vanilla(input_ids, attention_mask)
    actual, _ = tiny_gemma.run(input_ids, attention_mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_future_tokens_do_not_change_earlier_logits(tiny_gemma):
    tiny_gemma.alpha = 0.25
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8]])
    changed_ids = torch.tensor([[2, 3, 4, 5, 6, 20, 21]])
    mask = torch.ones_like(input_ids)
    original, _ = tiny_gemma.run(input_ids, mask)
    changed, _ = tiny_gemma.run(changed_ids, mask)
    # Valid for either first-pass or recirculated readout: no future-token leakage.
    torch.testing.assert_close(original[:, :5], changed[:, :5], rtol=1e-5, atol=1e-6)

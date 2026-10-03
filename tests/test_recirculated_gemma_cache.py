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


def test_run_requires_attention_mask(tiny_gemma):
    input_ids = torch.tensor([[2, 3]])
    with pytest.raises(TypeError, match="attention_mask"):
        tiny_gemma.run(input_ids)


def test_run_rejects_none_mask_before_computation(tiny_gemma, monkeypatch):
    def unexpected_embedding(*args, **kwargs):
        pytest.fail("Missing masks must be rejected before model computation")

    monkeypatch.setattr(tiny_gemma.model.model.embed_tokens, "forward", unexpected_embedding)
    with pytest.raises(ValueError, match="attention_mask must not be None"):
        tiny_gemma.run(torch.tensor([[2, 3]]), attention_mask=None)


@pytest.mark.parametrize("length", [1, 7])
@pytest.mark.parametrize("output_logits", ["first_pass", "recirculated_pass"])
def test_zero_alpha_matches_vanilla_across_sliding_window(tiny_gemma, length, output_logits):
    tiny_gemma.alpha = 0.0
    input_ids = torch.arange(2, 2 + length).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    expected, _ = tiny_gemma.run_vanilla(input_ids, attention_mask)
    actual, _ = tiny_gemma.run(input_ids, attention_mask, output_logits=output_logits)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("output_logits", ["first_pass", "recirculated_pass"])
def test_future_tokens_do_not_change_earlier_logits(tiny_gemma, output_logits):
    tiny_gemma.alpha = 0.25
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8]])
    changed_ids = torch.tensor([[2, 3, 4, 5, 6, 20, 21]])
    mask = torch.ones_like(input_ids)
    original, _ = tiny_gemma.run(input_ids, mask, output_logits=output_logits)
    changed, _ = tiny_gemma.run(changed_ids, mask, output_logits=output_logits)
    # Valid for either first-pass or recirculated readout: no future-token leakage.
    torch.testing.assert_close(original[:, :5], changed[:, :5], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("length", [1, 7])
def test_nonzero_alpha_selects_pass_outputs(tiny_gemma, monkeypatch, length):
    tiny_gemma.alpha = 0.25
    input_ids = torch.arange(2, 2 + length).unsqueeze(0)
    mask = torch.ones_like(input_ids)
    first_pass_states = []
    recirculated_states = []
    warmup_pass = tiny_gemma.warmup_pass
    block_pass = tiny_gemma.recirculated_block_pass
    final_pass = tiny_gemma.recirculate_token

    # Capture both readouts from real passes, before run selects its outputs.
    def capture_warmup(hidden_states):
        result = warmup_pass(hidden_states)
        first_pass_states.append(result.clone())
        return result

    def capture_pair(hidden_states, token_position):
        result = block_pass(hidden_states, token_position)
        recirculated_states.append(result[:, :1, :].clone())
        first_pass_states.append(result[:, 1:2, :].clone())
        return result

    def capture_final(token_position):
        result = final_pass(token_position)
        recirculated_states.append(result.clone())
        return result

    with monkeypatch.context() as hooks:
        hooks.setattr(tiny_gemma, "warmup_pass", capture_warmup)
        hooks.setattr(tiny_gemma, "recirculated_block_pass", capture_pair)
        hooks.setattr(tiny_gemma, "recirculate_token", capture_final)
        recirculated, _ = tiny_gemma.run(input_ids, mask, output_logits="recirculated_pass")

    with torch.no_grad():
        expected_first = tiny_gemma.model.lm_head(
            tiny_gemma.model.model.norm(torch.cat(first_pass_states, dim=1))
        )
        expected_recirculated = tiny_gemma.model.lm_head(
            tiny_gemma.model.model.norm(torch.cat(recirculated_states, dim=1))
        )

    first, _ = tiny_gemma.run(input_ids, mask, output_logits="first_pass")
    default, _ = tiny_gemma.run(input_ids, mask)
    assert first.shape == recirculated.shape == (1, length, tiny_gemma.model.config.vocab_size)
    # Compare all positions, including the warmup and final-token boundaries.
    torch.testing.assert_close(first, expected_first, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(recirculated, expected_recirculated, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(default, first, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(expected_first, expected_recirculated, rtol=1e-5, atol=1e-6)

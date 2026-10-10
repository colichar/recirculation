import importlib
from pathlib import Path

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig


execution_cases = pytest.mark.parametrize(
    "tiny_gemma",
    [("cpu", torch.float32), ("cuda", torch.float32), ("cuda", torch.bfloat16)],
    ids=["cpu-fp32", "cuda-fp32", "cuda-bf16"],
    indirect=True,
)


def assert_logits_close(actual, expected, *, rtol=1e-5, atol=1e-6):
    # Compare values in FP32 even if one forward path upcasts its logits.
    actual, expected = actual.float(), expected.float()
    difference = (actual - expected).abs()
    # Normalize by the global logit scale, not individual near-zero logits.
    scaled = difference.max() / expected.abs().max().clamp_min(torch.finfo(torch.float32).eps)
    print(f"max_abs={difference.max().item():.8g} mean_abs={difference.mean().item():.8g} "
          f"max_abs/scale={scaled.item():.8g}")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.fixture
def tiny_gemma(monkeypatch, request):
    device, dtype = getattr(request, "param", ("cpu", torch.float32))
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
            pytest.skip("CUDA device does not support native BF16")
        # Validate FP32 with IEEE matmul rather than TF32; restore after each test.
        monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
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
    def load_model(*args, **kwargs):
        assert kwargs["torch_dtype"] == dtype
        return model.to(device=device, dtype=kwargs["torch_dtype"])

    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", load_model)
    wrapper = module.RecirculatedGemma(source=2, destination=0, dtype=dtype)
    for parameter in wrapper.model.parameters():
        assert parameter.device.type == device
        assert parameter.dtype == dtype
    yield wrapper
    # Check actual execution storage, including noncommitting single-token passes.
    if wrapper.past_key_values is not None:
        for layer in wrapper.past_key_values.layers:
            for tensor in (layer.keys, layer.values):
                assert tensor.device.type == device
                assert tensor.dtype == dtype


def test_recirculated_run_requires_attention_mask(tiny_gemma):
    input_ids = torch.tensor([[2, 3]])
    with pytest.raises(TypeError, match="attention_mask"):
        tiny_gemma.recirculated_run(input_ids)


def test_recirculated_run_rejects_none_mask_before_computation(tiny_gemma, monkeypatch):
    def unexpected_embedding(*args, **kwargs):
        pytest.fail("Missing masks must be rejected before model computation")

    monkeypatch.setattr(tiny_gemma.model.model.embed_tokens, "forward", unexpected_embedding)
    with pytest.raises(ValueError, match="attention_mask must not be None"):
        tiny_gemma.recirculated_run(torch.tensor([[2, 3]]), attention_mask=None)


@execution_cases
@pytest.mark.parametrize("length", [1, 7, 13])
def test_zero_alpha_matches_vanilla_across_sliding_window(tiny_gemma, length):
    tiny_gemma.alpha = 0.0
    input_ids = torch.arange(2, 2 + length, device=tiny_gemma.model.device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    expected, _ = tiny_gemma.run_vanilla(input_ids, attention_mask)
    actual, _ = tiny_gemma.recirculated_run(input_ids, attention_mask)
    # BF16 allows approximately one rounding unit across different execution
    # schedules. Evidence and the tighter invariant tolerances are in
    # docs/numerical-validation.md.
    if tiny_gemma.model.dtype == torch.bfloat16:
        assert_logits_close(actual, expected, rtol=1e-2, atol=1e-5)
    else:
        assert_logits_close(actual, expected)


@execution_cases
def test_future_tokens_do_not_change_earlier_logits(tiny_gemma):
    tiny_gemma.alpha = 0.25
    input_ids = torch.arange(2, 15, device=tiny_gemma.model.device).unsqueeze(0)
    changed_ids = input_ids.clone()
    changed_ids[:, 11:] = torch.tensor([20, 21], device=input_ids.device)
    mask = torch.ones_like(input_ids)
    original, _ = tiny_gemma.recirculated_run(input_ids, mask)
    changed, _ = tiny_gemma.recirculated_run(changed_ids, mask)
    assert_logits_close(original[:, :11], changed[:, :11])


@execution_cases
@pytest.mark.parametrize("length", [1, 7])
def test_recirculated_run_returns_first_pass_readout(tiny_gemma, monkeypatch, length):
    tiny_gemma.alpha = 0.25
    input_ids = torch.arange(2, 2 + length, device=tiny_gemma.model.device).unsqueeze(0)
    mask = torch.ones_like(input_ids)
    first_pass_states = []
    warmup_pass = tiny_gemma.warmup_pass
    block_pass = tiny_gemma.recirculated_block_pass

    # Capture first-pass outputs from the real warmup and paired passes.
    def capture_warmup(**kwargs):
        result = warmup_pass(**kwargs)
        first_pass_states.append(result.clone())
        return result

    def capture_pair(**kwargs):
        result = block_pass(**kwargs)
        first_pass_states.append(result[:, 1:2, :].clone())
        return result

    with monkeypatch.context() as hooks:
        hooks.setattr(tiny_gemma, "warmup_pass", capture_warmup)
        hooks.setattr(tiny_gemma, "recirculated_block_pass", capture_pair)
        actual, _ = tiny_gemma.recirculated_run(input_ids, mask)

    with torch.no_grad():
        expected_first = tiny_gemma.model.lm_head(
            tiny_gemma.model.model.norm(torch.cat(first_pass_states, dim=1))
        )
    assert actual.shape == (1, length, tiny_gemma.model.config.vocab_size)
    # Compare all positions, including the warmup and paired-pass boundaries.
    assert_logits_close(actual, expected_first)

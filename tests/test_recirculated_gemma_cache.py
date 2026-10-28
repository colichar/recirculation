import importlib
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, Gemma3Config, Gemma3TextConfig, SiglipVisionConfig


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


@pytest.fixture(params=["text", "multimodal"], ids=["1b-layout", "4b-layout"])
def model_layout(request):
    return request.param


@pytest.fixture
def tiny_gemma(monkeypatch, request, model_layout):
    params = getattr(request, "param", ("cpu", torch.float32))
    device, dtype = params[:2]
    layer_types = params[2] if len(params) >= 3 else ["sliding_attention", "full_attention"] * 2
    alpha_ramp = params[3] if len(params) >= 4 else False
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
        layer_types=layer_types,
    )
    text_config = config
    if model_layout == "multimodal":
        config = Gemma3Config(
            text_config=text_config,
            vision_config=SiglipVisionConfig(
                hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, image_size=8, patch_size=2,
            ),
            mm_tokens_per_image=4,
            image_token_index=31,
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    # Exercise the real wrapper initialization without downloading weights or a tokenizer.
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: None)
    def load_model(*args, **kwargs):
        assert kwargs["torch_dtype"] == dtype
        return model.to(device=device, dtype=kwargs["torch_dtype"])

    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", load_model)
    # By default both stacks include sliding and full attention layers.
    wrapper_kwargs = {"source": 2, "destination": 1, "dtype": dtype, "alpha_ramp": alpha_ramp}
    if len(params) >= 5:
        wrapper_kwargs["beta"] = params[4]
    wrapper = module.RecirculatedGemma(**wrapper_kwargs)
    expected_decoder = model.model if model_layout == "text" else model.model.language_model
    assert wrapper.model_decoder is expected_decoder
    assert wrapper.model_config is text_config
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

    monkeypatch.setattr(tiny_gemma.model_decoder.embed_tokens, "forward", unexpected_embedding)
    with pytest.raises(ValueError, match="attention_mask must not be None"):
        tiny_gemma.recirculated_run(torch.tensor([[2, 3]]), attention_mask=None)


@execution_cases
@pytest.mark.parametrize("length", [1, 7, 13])
def test_zero_alpha_matches_vanilla_across_sliding_window(tiny_gemma, length):
    tiny_gemma.alpha = 0.0
    input_ids = torch.arange(2, 2 + length, device=tiny_gemma.model.device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    expected, _ = tiny_gemma.vanilla_run(input_ids, attention_mask)
    actual, _ = tiny_gemma.recirculated_run(input_ids, attention_mask)
    # BF16 allows approximately one rounding unit across different execution
    # schedules. Evidence and the tighter invariant tolerances are in
    # docs/numerical-validation.md.
    if tiny_gemma.model.dtype == torch.bfloat16:
        assert_logits_close(actual, expected, rtol=1e-2, atol=1e-5)
    else:
        assert_logits_close(actual, expected)


@pytest.mark.parametrize("model_layout", ["text"])
@pytest.mark.parametrize(
    "tiny_gemma",
    [
        (
            "cpu", torch.float32,
            ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
        )
    ],
    indirect=True,
    ids=["sliding-only-lower"],
)
def test_sliding_only_lower_stack_matches_vanilla_across_window(tiny_gemma):
    # Mirrors the 1B PT layout: no full-attention layer before the destination.
    assert tiny_gemma.lower_layers_attention == {"sliding_attention": 0}
    assert tiny_gemma.upper_layers_attention == {
        "full_attention": 2, "sliding_attention": 3
    }

    tiny_gemma.alpha = 0.0
    input_ids = torch.arange(2, 15, device=tiny_gemma.model.device).unsqueeze(0)
    mask = torch.ones_like(input_ids)
    expected, _ = tiny_gemma.vanilla_run(input_ids, mask)
    actual, _ = tiny_gemma.recirculated_run(input_ids, mask)
    assert_logits_close(actual, expected)

    # Lower-layer KV was committed for every token; upper KV trails by one.
    state = snapshot_generation_state(tiny_gemma)
    assert state["lengths"] == [13, 13, 12, 12]
    assert state["positions"] == state["lengths"]


@pytest.mark.parametrize("model_layout", ["text"])
@pytest.mark.parametrize(
    "tiny_gemma",
    [
        (
            "cpu", torch.float32,
            ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
            True,
        )
    ],
    indirect=True,
    ids=["sliding-only-lower-ramp"],
)
def test_alpha_ramp_resets_for_each_window_and_preserves_constant_mode(tiny_gemma, monkeypatch):
    tiny_gemma.alpha = 0.07
    alphas = []
    block_pass = tiny_gemma.recirculated_block_pass

    def capture_alpha(**kwargs):
        alphas.append(kwargs["alpha"])
        return block_pass(**kwargs)

    monkeypatch.setattr(tiny_gemma, "recirculated_block_pass", capture_alpha)
    ids = torch.arange(2, 15).unsqueeze(0)
    ramped_logits, _ = tiny_gemma.recirculated_run(ids, torch.ones_like(ids))
    assert len(alphas) == 12
    assert [alphas[i] for i in (0, 1, 9, 10, 11)] == pytest.approx(
        [0.0, 0.007, 0.063, 0.07, 0.07]
    )

    alphas.clear()
    next_window = torch.tensor([[7, 8, 9]])
    tiny_gemma.recirculated_run(next_window, torch.ones_like(next_window))
    assert alphas == pytest.approx([0.0, 0.007])

    alphas.clear()
    tiny_gemma.alpha_ramp = False
    constant_logits, _ = tiny_gemma.recirculated_run(ids, torch.ones_like(ids))
    assert alphas == pytest.approx([0.07] * 12)
    assert not torch.allclose(ramped_logits, constant_logits, rtol=1e-5, atol=1e-6)

    tiny_gemma.alpha_ramp = True
    tiny_gemma.alpha = 0.0
    zero_logits, _ = tiny_gemma.recirculated_run(ids, torch.ones_like(ids))
    vanilla_logits, _ = tiny_gemma.vanilla_run(ids, torch.ones_like(ids))
    assert_logits_close(zero_logits, vanilla_logits)
    assert not torch.allclose(ramped_logits, zero_logits, rtol=1e-5, atol=1e-6)


def test_alpha_ramp_continues_from_prompt_during_generation(tiny_gemma, monkeypatch):
    tiny_gemma.alpha = 0.07
    tiny_gemma.alpha_ramp = True
    alphas = []
    block_pass = tiny_gemma.recirculated_block_pass

    def capture_alpha(**kwargs):
        alphas.append(kwargs["alpha"])
        return block_pass(**kwargs)

    monkeypatch.setattr(tiny_gemma, "recirculated_block_pass", capture_alpha)
    ids = torch.tensor([[2, 3, 4, 5]])
    tiny_gemma.recirculated_generate(ids, torch.ones_like(ids), max_new_tokens=2)
    # Prompt recirculates positions 0..2; decoding next recirculates position 3.
    assert alphas == pytest.approx([0.0, 0.007, 0.014, 0.021])


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
def test_right_padding_preserves_valid_logits(tiny_gemma):
    tiny_gemma.alpha = 0.09
    ids = torch.arange(2, 9, device=tiny_gemma.model.device).unsqueeze(0)
    mask = torch.ones_like(ids)
    expected, _ = tiny_gemma.recirculated_run(ids, mask)
    pad_mask = torch.cat([mask, torch.zeros_like(mask[:, :2])], dim=1)
    for padding in ([0, 0], [20, 21]):
        padded_ids = torch.cat([ids, ids.new_tensor([padding])], dim=1)
        actual, _ = tiny_gemma.recirculated_run(padded_ids, pad_mask)
        assert_logits_close(actual[:, :ids.shape[1]], expected)


class ReferenceHistory:
    """Untruncated KV history for single-token reference passes, without cache helpers."""

    def __init__(self, num_layers):
        self.kv = [None] * num_layers
        self.commit = False

    def update(self, keys, values, layer_idx):
        if self.kv[layer_idx] is None:
            self.kv[layer_idx] = (keys[..., :0, :], values[..., :0, :])
        old_keys, old_values = self.kv[layer_idx]
        full = (
            torch.cat([old_keys, keys], dim=-2),
            torch.cat([old_values, values], dim=-2),
        )
        if self.commit:
            self.kv[layer_idx] = full
        return full


@torch.no_grad()
def sequential_recirculation_reference(wrapper, input_ids, attention_mask):
    """K=1, constant alpha: read a token, then commit its separate recirculation.

    Shares only the model's decoder modules with the implementation. All history
    is retained; absolute-position masks enforce sliding attention independently
    of cache eviction and the paired-pass mask/position calculations.
    """
    assert not wrapper.alpha_ramp
    decoder = wrapper.model_decoder
    config = wrapper.model_config
    history = ReferenceHistory(config.num_hidden_layers)
    first_pass_states = []

    def run_layers(hidden, layers, position, commit):
        history.commit = commit
        position_ids = input_ids.new_tensor([[position]])
        key_positions = torch.arange(position + 1, device=input_ids.device)
        source = None
        for layer_idx in layers:
            layer_type = config.layer_types[layer_idx]
            allowed = (key_positions <= position) & attention_mask[:, :position + 1].bool()
            if layer_type == "sliding_attention":
                allowed = allowed & (key_positions > position - config.sliding_window)
            mask = hidden.new_zeros((input_ids.shape[0], 1, 1, position + 1))
            mask.masked_fill_(~allowed[:, None, None, :], torch.finfo(hidden.dtype).min)
            hidden = decoder.layers[layer_idx](
                hidden_states=hidden,
                position_embeddings=decoder.rotary_emb(hidden, position_ids, layer_type),
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=history,
            )
            if layer_idx == wrapper.source:
                source = hidden
        return hidden, source

    lower = range(wrapper.destination + 1)
    upper = range(wrapper.destination + 1, config.num_hidden_layers)
    for position in range(input_ids.shape[1]):
        embedded = decoder.embed_tokens(input_ids[:, position:position + 1])
        destination, _ = run_layers(embedded, lower, position, commit=True)
        first_pass, source = run_layers(destination, upper, position, commit=False)
        first_pass_states.append(first_pass)
        # Match the scoring boundary: the last token's upper KV stays uncommitted.
        if position + 1 < input_ids.shape[1]:
            normalized_source = source * (destination.norm(dim=-1, keepdim=True)
                                          / source.norm(dim=-1, keepdim=True))
            mixed = wrapper.beta * destination + wrapper.alpha * normalized_source
            run_layers(mixed, upper, position, commit=True)

    logits = wrapper.model.lm_head(decoder.norm(torch.cat(first_pass_states, dim=1)))
    return logits, history


@execution_cases
@pytest.mark.parametrize("length,padding", [(1, 0), (7, 0), (13, 2)])
@pytest.mark.parametrize("alpha", [0.0, 0.09])
def test_run_matches_sequential_reference_and_retained_kv(tiny_gemma, length, padding, alpha):
    tiny_gemma.alpha = alpha
    assert tiny_gemma.beta == 1
    ids = torch.arange(2, 2 + length, device=tiny_gemma.model.device).unsqueeze(0)
    mask = torch.ones_like(ids)
    if padding:
        # Check padded-query logits too: a valid-prefix-only check cannot detect
        # ignored right-padding masks, since causality already excludes future PAD.
        ids = torch.cat([ids, ids.new_zeros((1, padding))], dim=1)
        mask = torch.cat([mask, mask.new_zeros((1, padding))], dim=1)
    expected, history = sequential_recirculation_reference(tiny_gemma, ids, mask)
    actual, _ = tiny_gemma.recirculated_run(ids, mask)
    rtol, atol = (1e-2, 1e-5) if tiny_gemma.model.dtype == torch.bfloat16 else (1e-5, 1e-6)
    assert_logits_close(actual, expected, rtol=rtol, atol=atol)
    if alpha == 0:
        vanilla, _ = tiny_gemma.vanilla_run(ids, mask)
        assert_logits_close(expected, vanilla, rtol=rtol, atol=atol)

    config = tiny_gemma.model_config
    cache = tiny_gemma.past_key_values
    for layer_idx, layer in enumerate(cache.layers):
        committed = ids.shape[1] if layer_idx <= tiny_gemma.destination else ids.shape[1] - 1
        assert cache.get_seq_length(layer_idx) == committed
        assert cache.get_query_offset(layer_idx) == committed
        sliding = config.layer_types[layer_idx] == "sliding_attention"
        retained = min(committed, config.sliding_window - 1) if sliding else committed
        for actual_kv, reference_kv in zip((layer.keys, layer.values), history.kv[layer_idx], strict=True):
            assert reference_kv.shape[-2] == committed
            if retained == 0:
                # HF initializes full-attention storage as a 1-D empty tensor.
                assert actual_kv.numel() == 0
                continue
            assert actual_kv.shape[-2] == retained
            # Separate single-token and paired matmuls differ slightly after key
            # normalization (observed CPU FP32 max_abs=1.49e-6, key scale=2.01).
            kv_atol = 3e-6 if tiny_gemma.model.dtype == torch.float32 else atol
            torch.testing.assert_close(
                actual_kv, reference_kv[..., committed - retained:, :], rtol=rtol, atol=kv_atol
            )


@pytest.mark.parametrize(
    "tiny_gemma",
    [
        ("cpu", torch.float32, ["sliding_attention", "full_attention"] * 2, False, 0.93),
        ("cuda", torch.bfloat16, ["sliding_attention", "full_attention"] * 2, False, 0.93),
    ],
    ids=["cpu-fp32", "cuda-bf16"],
    indirect=True,
)
def test_nonunit_beta_matches_sequential_reference(tiny_gemma):
    assert tiny_gemma.beta == 0.93
    tiny_gemma.alpha = 0.07
    ids = torch.arange(2, 15, device=tiny_gemma.model.device).unsqueeze(0)
    mask = torch.ones_like(ids)

    expected, _ = sequential_recirculation_reference(tiny_gemma, ids, mask)
    actual, _ = tiny_gemma.recirculated_run(ids, mask)

    rtol, atol = (1e-2, 1e-5) if tiny_gemma.model.dtype == torch.bfloat16 else (1e-5, 1e-6)
    assert_logits_close(actual, expected, rtol=rtol, atol=atol)


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
            tiny_gemma.model_decoder.norm(torch.cat(first_pass_states, dim=1))
        )
    assert actual.shape == (1, length, tiny_gemma.model_config.vocab_size)
    # Compare all positions, including the warmup and paired-pass boundaries.
    assert_logits_close(actual, expected_first)


def greedy_generation_reference(input_ids, attention_mask, max_new_tokens, forward):
    logits_per_step = []
    generated_ids = []

    for _ in range(max_new_tokens):
        logits, _ = forward(input_ids, attention_mask)
        next_logits = logits[:, -1, :]
        next_token_id = next_logits.argmax(dim=-1, keepdim=True)
        logits_per_step.append(next_logits)
        generated_ids.append(next_token_id)

        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token_id)], dim=1
        )

    return torch.stack(logits_per_step, dim=1), torch.cat(generated_ids, dim=1)


def snapshot_generation_state(wrapper):
    cache = wrapper.past_key_values
    return {
        "lengths": [cache.get_seq_length(i) for i in range(len(cache.layers))],
        "positions": [cache.get_query_offset(i) for i in range(len(cache.layers))],
        "kv": [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers],
        "destination": wrapper.destination_hidden.clone(),
        "source": wrapper.source_hidden.clone(),
    }


def assert_generation_state_matches_reference(actual, expected, wrapper, processed_length):
    assert actual["lengths"] == expected["lengths"] == [
        processed_length if i <= wrapper.destination else processed_length - 1
        for i in range(wrapper.model_config.num_hidden_layers)
    ]
    assert actual["positions"] == expected["positions"] == actual["lengths"]
    rtol, atol = (1e-2, 1e-5) if wrapper.model.dtype == torch.bfloat16 else (1e-5, 1e-6)
    for actual_kv, expected_kv in zip(actual["kv"], expected["kv"], strict=True):
        for actual_tensor, expected_tensor in zip(actual_kv, expected_kv, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual["destination"], expected["destination"], rtol=rtol, atol=atol)
    torch.testing.assert_close(actual["source"], expected["source"], rtol=rtol, atol=atol)


@execution_cases
@pytest.mark.parametrize("prompt_length", [1, 4])
def test_recirculated_generate_matches_full_sequence_reference(
    tiny_gemma, monkeypatch, prompt_length
):
    tiny_gemma.alpha = 0.25
    input_ids = torch.arange(
        2, 2 + prompt_length, device=tiny_gemma.model.device
    ).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    max_new_tokens = 3

    def unexpected_standalone_recirculation(*args, **kwargs):
        pytest.fail("first-pass generation must not recirculate the final token alone")

    monkeypatch.setattr(
        tiny_gemma, "recirculate_token", unexpected_standalone_recirculation
    )
    actual_logits, actual_ids = tiny_gemma.recirculated_generate(
        input_ids, attention_mask, max_new_tokens=max_new_tokens
    )
    actual_state = snapshot_generation_state(tiny_gemma)

    expected_logits, expected_ids = greedy_generation_reference(
        input_ids.clone(),
        attention_mask.clone(),
        max_new_tokens,
        tiny_gemma.recirculated_run,
    )
    assert_logits_close(actual_logits, expected_logits)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)

    # The last sampled ID has not been processed. The preceding generated token
    # has first-pass source/destination states, but uncommitted upper-layer KV.
    processed_ids = torch.cat([input_ids, actual_ids[:, :-1]], dim=1)
    tiny_gemma.recirculated_run(processed_ids, torch.ones_like(processed_ids))
    expected_state = snapshot_generation_state(tiny_gemma)
    assert_generation_state_matches_reference(
        actual_state, expected_state, tiny_gemma, processed_ids.shape[1]
    )


@execution_cases
@pytest.mark.parametrize("max_new_tokens", [1, 2])
def test_generation_prefill_and_first_decoding_transition(
    tiny_gemma, monkeypatch, max_new_tokens
):
    input_ids = torch.tensor([[2, 3, 4, 5]], device=tiny_gemma.model.device)
    attention_mask = torch.ones_like(input_ids)
    lower_positions = {layer: [] for layer in range(tiny_gemma.destination + 1)}
    cache_initializations = []
    module = importlib.import_module("recirculated_gemma")
    cache_class = module.RecirculationDynamicCache

    def capture_lower_forward(layer_idx, forward):
        def capture(**kwargs):
            lower_positions[layer_idx].append(kwargs["position_ids"].clone())
            return forward(**kwargs)

        return capture

    def create_cache(**kwargs):
        cache_initializations.append(1)
        return cache_class(**kwargs)

    with monkeypatch.context() as hooks:
        for layer_idx in lower_positions:
            layer = tiny_gemma.model_decoder.layers[layer_idx]
            hooks.setattr(layer, "forward", capture_lower_forward(layer_idx, layer.forward))
        hooks.setattr(module, "RecirculationDynamicCache", create_cache)
        _, generated_ids = tiny_gemma.recirculated_generate(
            input_ids, attention_mask, max_new_tokens=max_new_tokens
        )
    actual_state = snapshot_generation_state(tiny_gemma)

    assert len(cache_initializations) == 1
    for positions in lower_positions.values():
        assert len(positions) == max_new_tokens
        torch.testing.assert_close(
            positions[0], torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        )
        if max_new_tokens == 2:
            torch.testing.assert_close(
                positions[1], torch.tensor([[input_ids.shape[1]]], device=input_ids.device)
            )

    processed_ids = torch.cat([input_ids, generated_ids[:, :-1]], dim=1)
    tiny_gemma.recirculated_run(processed_ids, torch.ones_like(processed_ids))
    expected_state = snapshot_generation_state(tiny_gemma)
    assert_generation_state_matches_reference(
        actual_state, expected_state, tiny_gemma, processed_ids.shape[1]
    )


@execution_cases
def test_nonzero_alpha_changes_run_and_generation_logits(tiny_gemma):
    input_ids = torch.arange(2, 9, device=tiny_gemma.model.device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    tiny_gemma.alpha = 0.0
    zero_alpha_run, _ = tiny_gemma.recirculated_run(input_ids, attention_mask)
    zero_alpha_generate, _ = tiny_gemma.recirculated_generate(
        input_ids, attention_mask, max_new_tokens=3
    )

    tiny_gemma.alpha = 0.25
    nonzero_alpha_run, _ = tiny_gemma.recirculated_run(input_ids, attention_mask)
    nonzero_alpha_generate, _ = tiny_gemma.recirculated_generate(
        input_ids, attention_mask, max_new_tokens=3
    )

    assert not torch.allclose(zero_alpha_run, nonzero_alpha_run, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(
        zero_alpha_generate, nonzero_alpha_generate, rtol=1e-5, atol=1e-6
    )


@execution_cases
@pytest.mark.parametrize("prompt_length", [1, 4])
def test_zero_alpha_generate_matches_vanilla_reference(tiny_gemma, prompt_length):
    tiny_gemma.alpha = 0.0
    input_ids = torch.arange(
        2, 2 + prompt_length, device=tiny_gemma.model.device
    ).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    max_new_tokens = 3

    actual_logits, actual_ids = tiny_gemma.recirculated_generate(
        input_ids, attention_mask, max_new_tokens=max_new_tokens
    )
    expected_logits, expected_ids = greedy_generation_reference(
        input_ids.clone(),
        attention_mask.clone(),
        max_new_tokens,
        tiny_gemma.vanilla_run,
    )

    if tiny_gemma.model.dtype == torch.bfloat16:
        assert_logits_close(actual_logits, expected_logits, rtol=1e-2, atol=1e-5)
    else:
        assert_logits_close(actual_logits, expected_logits)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)


@execution_cases
def test_recirculated_generate_resets_state_between_requests(tiny_gemma):
    first_prompt = torch.tensor([[2, 3, 4, 5]], device=tiny_gemma.model.device)
    other_prompt = torch.tensor([[6, 7, 8, 9, 10]], device=tiny_gemma.model.device)
    tiny_gemma.alpha = 0.25

    first_logits, first_ids = tiny_gemma.recirculated_generate(
        first_prompt, torch.ones_like(first_prompt), max_new_tokens=3
    )
    tiny_gemma.recirculated_generate(
        other_prompt, torch.ones_like(other_prompt), max_new_tokens=2
    )
    second_logits, second_ids = tiny_gemma.recirculated_generate(
        first_prompt, torch.ones_like(first_prompt), max_new_tokens=3
    )

    assert_logits_close(first_logits, second_logits)
    torch.testing.assert_close(first_ids, second_ids, rtol=0, atol=0)

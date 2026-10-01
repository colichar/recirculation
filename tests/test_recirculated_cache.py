import pytest
import torch
import torch.nn.functional as F
from transformers import Gemma3TextConfig
from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from recirculation.recirculated_cache import (
    RecirculationDynamicCache,
    RecirculationDynamicLayer,
    RecirculationDynamicSlidingWindowLayer,
)


def token_kv(tokens):
    """Identifiable tokens with distinct batch, head, feature, and value contents."""
    tokens = torch.tensor(tokens, dtype=torch.float32).reshape(1, 1, -1, 1)
    offsets = torch.arange(12, dtype=torch.float32).reshape(2, 2, 1, 3)
    keys = 100 * tokens + offsets
    return keys, -keys - 1


def assert_kv_equal(actual, expected):
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@pytest.fixture(params=["full", "sliding", "sliding_recording"])
def layer_pair(request):
    if request.param == "full":
        return RecirculationDynamicLayer(), DynamicLayer()
    custom = RecirculationDynamicSlidingWindowLayer(sliding_window=4)
    vanilla = DynamicSlidingWindowLayer(sliding_window=4)
    if request.param == "sliding_recording":
        custom.activate_past_recording()
        vanilla.activate_past_recording()
    return custom, vanilla


def test_commit_none_matches_transformers_prefill_and_append(layer_pair):
    custom, vanilla = layer_pair
    for tokens in ([0, 1, 2, 3, 4, 5], [6, 7]):
        keys, values = token_kv(tokens)
        assert_kv_equal(
            custom.update(keys, values, commit_tokens=None),
            vanilla.update(keys, values),
        )
        assert_kv_equal((custom.keys, custom.values), (vanilla.keys, vanilla.values))
        assert custom.get_seq_length() == vanilla.get_seq_length()


# K=1 is the paper's schedule; K=2 checks the cache API for blockwise experiments.
@pytest.mark.parametrize("block_size", [1, 2])
def test_warmup_and_paired_passes_commit_only_recirculated_prefix(layer_pair, block_size):
    layer, _ = layer_pair
    warmup = token_kv(list(range(block_size)))
    assert_kv_equal(layer.update(*warmup, commit_tokens=0), warmup)
    assert layer.get_seq_length() == 0
    assert layer.keys.numel() == layer.values.numel() == 0

    committed_tokens = []
    for start in range(0, 8, block_size):
        # A token's recirculated KV differs from its discarded first-pass KV.
        recirculated = [1000 + t for t in range(start, start + block_size)]
        first_pass = list(range(start + block_size, start + 2 * block_size))
        stored_tokens = committed_tokens
        if layer.is_sliding and not layer.record_past:
            stored_tokens = stored_tokens[-3:]

        if not layer.is_sliding or not layer.record_past:
            assert layer.get_mask_sizes(2 * block_size) == (
                len(stored_tokens) + 2 * block_size,
                len(committed_tokens) - len(stored_tokens),
            )

        attention_kv = layer.update(*token_kv(recirculated + first_pass), commit_tokens=block_size)
        assert_kv_equal(attention_kv, token_kv(stored_tokens + recirculated + first_pass))

        committed_tokens.extend(recirculated)
        expected_stored = committed_tokens
        if layer.is_sliding and not layer.record_past:
            expected_stored = expected_stored[-3:]
        assert_kv_equal((layer.keys, layer.values), token_kv(expected_stored))
        assert layer.get_seq_length() == len(committed_tokens)

    # A noncommitting call with existing history must neither append nor evict.
    before = (layer.keys.clone(), layer.values.clone())
    incoming = token_kv([99])
    expected_attention = tuple(torch.cat([old, new], dim=-2) for old, new in zip(before, incoming))
    assert_kv_equal(layer.update(*incoming, commit_tokens=0), expected_attention)
    assert_kv_equal((layer.keys, layer.values), before)
    assert layer.get_seq_length() == len(committed_tokens)


@pytest.fixture
def mixed_config():
    config = Gemma3TextConfig(
        num_hidden_layers=4,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=4,
    )
    config._attn_implementation = "eager"
    return config


def test_configured_cache_keeps_lower_and_upper_layer_histories_independent(mixed_config):
    cache = RecirculationDynamicCache(config=mixed_config)
    assert [type(layer) for layer in cache.layers] == [
        RecirculationDynamicSlidingWindowLayer,
        RecirculationDynamicLayer,
    ] * 2

    # Lower layers have already processed the entire prompt.
    for layer_idx in (0, 1):
        cache.update(*token_kv(list(range(7))), layer_idx=layer_idx)
    lower_kv = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers[:2]]

    cache.commit_tokens = 0
    for layer_idx in (2, 3):
        assert_kv_equal(cache.update(*token_kv([0]), layer_idx=layer_idx), token_kv([0]))
        assert cache.get_seq_length(layer_idx) == 0

    cache.commit_tokens = 1
    for token in range(5):
        for layer_idx in (2, 3):
            cache.update(*token_kv([1000 + token, token + 1]), layer_idx=layer_idx)
            assert cache.get_seq_length(layer_idx) == token + 1
            assert cache.get_query_offset(layer_idx) == token + 1

    assert_kv_equal((cache.layers[2].keys, cache.layers[2].values), token_kv([1002, 1003, 1004]))
    assert_kv_equal((cache.layers[3].keys, cache.layers[3].values), token_kv(list(range(1000, 1005))))
    for layer_idx in (0, 1):
        assert cache.get_seq_length(layer_idx) == 7
        assert_kv_equal((cache.layers[layer_idx].keys, cache.layers[layer_idx].values), lower_kv[layer_idx])


@pytest.mark.parametrize("layer_idx", [2, 3], ids=["sliding", "full"])
def test_cached_attention_matches_absolute_position_history_reference(mixed_config, layer_idx):
    """Compare real HF masks/cache with an untruncated, explicitly masked history."""
    generator = torch.Generator().manual_seed(17)
    cache = RecirculationDynamicCache(config=mixed_config)
    # A much longer lower-layer cache must not shift upper-layer query positions.
    for lower_idx in (0, 1):
        cache.update(*token_kv(list(range(12))), layer_idx=lower_idx)

    sliding = mixed_config.layer_types[layer_idx] == "sliding_attention"
    make_mask = create_sliding_window_causal_mask if sliding else create_causal_mask
    history_keys = torch.empty(2, 2, 0, 3)
    history_values = torch.empty_like(history_keys)

    for step in range(9):
        query_length = 1 if step == 0 else 2
        cache.commit_tokens = 0 if step == 0 else 1
        position = history_keys.shape[-2]
        queries, keys, values = [
            torch.randn(2, 2, query_length, 3, generator=generator) for _ in range(3)
        ]
        positions = torch.arange(position, position + query_length)
        mask = make_mask(
            config=mixed_config,
            inputs_embeds=torch.zeros(2, query_length, 6),
            attention_mask=torch.ones(2, position + query_length),
            past_key_values=cache,
            position_ids=positions.unsqueeze(0),
            layer_idx=layer_idx,
        )
        attention_keys, attention_values = cache.update(keys, values, layer_idx=layer_idx)
        assert mask.shape[-2:] == (query_length, attention_keys.shape[-2])
        actual = F.scaled_dot_product_attention(queries, attention_keys, attention_values, attn_mask=mask)

        # Reference retains all committed tokens, independently of cache eviction
        # and HF's mask-size/offset calculations. Only absolute positions decide visibility.
        reference_keys = torch.cat([history_keys, keys], dim=-2)
        reference_values = torch.cat([history_values, values], dim=-2)
        key_positions = torch.arange(reference_keys.shape[-2])
        allowed = key_positions[None, :] <= positions[:, None]
        if sliding:
            allowed &= key_positions[None, :] > positions[:, None] - mixed_config.sliding_window
        scores = queries @ reference_keys.transpose(-1, -2) / queries.shape[-1] ** 0.5
        expected = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1) @ reference_values
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        if step > 0:
            history_keys = torch.cat([history_keys, keys[..., :1, :]], dim=-2)
            history_values = torch.cat([history_values, values[..., :1, :]], dim=-2)

import torch
from transformers.cache_utils import (
    DynamicLayer,
    DynamicSlidingWindowLayer,
    Cache,
    get_layer_types_and_kwargs
)
from collections.abc import Iterable
from transformers import PreTrainedConfig

class RecirculationDynamicLayer(DynamicLayer):
    def __init__(self, **kwargs):
        super().__init__()

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, commit_tokens: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return KV for current attention and retain only committed tokens.
        Cases:
            - commit_tokens == None:            Prefill warmup lower layers forward pass for all tokens.
            - commit_tokens == 0:               Noncommitting pass, including initial upper-layer warmup.
            - commit_tokens == block size K:    2*K token forward pass, with first K tokens being recirculated
                                                and the other K tokens being warmed up.
        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            commit_tokens (`int`, optional): None commits all incoming tokens; 0 commits none.
                Positive K is supported only for paired passes with exactly 2*K incoming tokens:
                commit the first K recirculated tokens and leave the next K uncommitted.
                Standalone final-token commits are outside the project scope.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: Previously stored KV plus all incoming KV,
                including uncommitted tokens needed for current attention.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        attention_keys = torch.cat([self.keys, key_states], dim=-2)
        attention_values = torch.cat([self.values, value_states], dim=-2)

        if commit_tokens is None:
            self.keys = attention_keys
            self.values = attention_values

        elif commit_tokens == 0:
            pass

        else:
            self.keys = torch.cat([self.keys, key_states[..., :commit_tokens, :]], dim=-2)
            self.values = torch.cat([self.values, value_states[..., :commit_tokens, :]], dim=-2)

        return attention_keys, attention_values


class RecirculationDynamicSlidingWindowLayer(DynamicSlidingWindowLayer):
    def __init__(self, sliding_window: int, **kwargs):
        super().__init__(sliding_window=sliding_window, **kwargs)

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, commit_tokens: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return KV for current attention and retain committed KV within the sliding window.

        Retain at most sliding_window - 1 committed tokens, or all committed tokens when
        recording past states until crop is called.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            commit_tokens (`int`, optional): None commits all incoming tokens; 0 commits none.
                Positive K is supported only for paired passes with exactly 2*K incoming tokens:
                commit the first K recirculated tokens and leave the next K uncommitted.
                Standalone final-token commits are outside the project scope.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: Previously stored KV plus all incoming KV,
                before window eviction and including uncommitted tokens for current attention.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Compute the full states
        full_key_states = torch.cat([self.keys, key_states], dim=-2)
        full_value_states = torch.cat([self.values, value_states], dim=-2)
        # Only cache the last `self.sliding_window - 1` tokens (or all of them if lower than that)
        if not self.record_past:
            if commit_tokens is None:
                self.cumulative_length += key_states.shape[-2]

                self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
                self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
            elif commit_tokens == 0:
                pass
            else:
                self.cumulative_length += commit_tokens

                self.keys = full_key_states[..., -self.sliding_window - commit_tokens + 1 : - commit_tokens, :]
                self.values = full_value_states[..., -self.sliding_window - commit_tokens + 1 : - commit_tokens, :]

        # If we record the past, we keep them all for now, and they'll be restricted to the window size in `crop`
        else:
            if commit_tokens is None:
                self.cumulative_length += key_states.shape[-2]

                self.keys = full_key_states
                self.values = full_value_states
            elif commit_tokens == 0:
                pass
            else:
                self.cumulative_length += commit_tokens

                self.keys = full_key_states[..., : -commit_tokens, :]
                self.values = full_value_states[..., : - commit_tokens, :]

        # Return the full states
        return full_key_states, full_value_states


RECIRCULATION_LAYER_TYPE_MAPPING = {
    "full_attention": RecirculationDynamicLayer,
    "sliding_attention": RecirculationDynamicSlidingWindowLayer,
}

class RecirculationDynamicCache(Cache):
    """Cache for the project's paired-pass Recirculation schedule.

    Always construct this cache with a model config. Config-free initialization is
    outside the project scope, despite the optional constructor signature.

    Set commit_tokens to None to commit all incoming tokens, or 0 to commit none.
    Partial commits use exactly 2*K incoming tokens: the first K are recirculated
    and committed, while the next K are warmed up and remain uncommitted.
    Standalone final-token commits are outside the project scope.
    """

    def __init__(
        self,
        ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
        config: PreTrainedConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        self.commit_tokens = None
        if offloading:
            raise NotImplementedError(
                "Cache offloading is not supported by RecirculationDynamicCache."
            )

        layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            layer_types, layer_kwargs = get_layer_types_and_kwargs(decoder_config)
            # Dispatch the layer types
            layers = [RECIRCULATION_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs) for layer_type in layer_types]

        # In this case, use the passed data to already fill in the Cache
        if ddp_cache_data is not None:
            # Init all the layers with the data
            for layer_idx, kv_and_optional_sliding in enumerate(ddp_cache_data):
                # kv_and_optional_sliding contains at least two elements: the key and value states. It can also
                # contain a third element, which is an optional sliding window tensor.
                key_states, value_states = kv_and_optional_sliding[:2]
                sliding_window_tensor = kv_and_optional_sliding[2] if len(kv_and_optional_sliding) == 3 else None
                # If the config was not passed above, initialize a new cache layer for each entry of the ddp_data
                if config is None:
                    # If there is a sliding window tensor, use it to initialize the layer
                    if sliding_window_tensor is not None:
                        layers.append(RecirculationDynamicSlidingWindowLayer(sliding_window=sliding_window_tensor.item()))
                    else:
                        layers.append(RecirculationDynamicLayer())
                # Update the layer with the data if any
                if key_states is not None and value_states is not None:
                    _, _ = layers[layer_idx].update(key_states, value_states)

        # Inherited constructor fallback; config-free use is outside the supported scope.
        if len(layers) == 0:
            super().__init__(
                layer_class_to_replicate=RecirculationDynamicLayer,
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding,
            )
        else:
            super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, getattr(layer, "_sliding_window_tensor", None)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        # Cache offloading currently unsupported
        return self.layers[layer_idx].update(
            key_states=key_states,
            value_states=value_states,
            commit_tokens=self.commit_tokens
        )

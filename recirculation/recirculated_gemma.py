import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.masking_utils import (create_causal_mask, create_sliding_window_causal_mask)
from recirculation.recirculated_cache import RecirculationDynamicCache


class RecirculatedGemma:
    def __init__(
            self,
            source=11,
            destination=4,
            alpha=0.25,
            alpha_ramp=False,
            model_name="google/gemma-3-1b-it",
            model_revision="fcf18a2a879aab110ca39f8bffbccd5d49d8eb29",
            dtype: torch.dtype = torch.float32
            ):
        # Evaluation relevant parameters
        self.source = source
        self.destination = destination
        self.alpha = alpha
        self.alpha_ramp = alpha_ramp
        # Paper used beta = 1 for all experiments
        self.beta = 1

        # Loading the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
        self.model = AutoModelForCausalLM.from_pretrained( 
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            revision=model_revision
        )

        self.model_config = self.model.config.get_text_config(decoder=True)
        self.model_decoder = self.model.get_decoder()

        self.model.eval()

        self.past_key_values = None
        self.destination_hidden = None
        self.source_hidden = None

        self.final_hidden_states = []

        lower_layers = range(self.destination + 1)
        upper_layers = range(
            self.destination + 1,
            self.model_config.num_hidden_layers
        )

        self.lower_layers_attention = self._create_layer_attention_mappings(layer_range=lower_layers)
        self.upper_layers_attention = self._create_layer_attention_mappings(layer_range=upper_layers)

    def _create_layer_attention_mappings(self, layer_range):
        mapping = {}
        for i in layer_range:
            mapping.setdefault(self.model_config.layer_types[i], i)

        return mapping

    def _compute_effective_alpha(self, token_position):
        if self.alpha_ramp:
            return min(token_position/10, 1) * self.alpha

        return self.alpha

    def recirculated_run(self,
            input_ids,
            attention_mask,
            reset_cache=True,
            input_embeddings=None,
            position_ids=None,
            position_embeddings=None,
            K=1
            ):
        """Return per-token logits and input IDs, resetting the cache each call.

        attention_mask is required and must not be None. Pass a mask of shape
        (batch_size, sequence_length), with 1 for valid tokens and 0 for padding;
        use an all-ones mask for unpadded inputs. Causal and sliding-window
        restrictions are applied internally.
        """
        if attention_mask is None:
            raise ValueError("attention_mask must not be None; use an all-ones mask for unpadded inputs.")

        with torch.no_grad():
            self.final_hidden_states = []

            if reset_cache:
                ## Empty Cache
                self.past_key_values = RecirculationDynamicCache(config=self.model_config)

            if input_embeddings is None:
                ## Generate Embeddings
                input_embeddings = self.generate_input_embeddigs(input_ids=input_ids)

            if position_ids is None or position_embeddings is None:
                position_ids, position_embeddings = self.generate_position_embeddings(hidden_states=input_embeddings)

            hidden_states = input_embeddings.clone()

            # Warmup all tokens
            self.past_key_values.commit_tokens = None

            causal_mask_mapping = self.create_attention_masks(
                input_embeddings=hidden_states,
                attention_mask=attention_mask,
                token_position=None,
                past_key_values=self.past_key_values,
                position_ids=position_ids,
                mask_layer_indices=self.lower_layers_attention
            )

            for layer in range(self.destination + 1):
                layer_type = self.model_config.layer_types[layer]
                    
                hidden_states = self.model_decoder.layers[layer].forward(
                    hidden_states=hidden_states,
                    position_embeddings=(position_embeddings[layer_type]["cos"], position_embeddings[layer_type]["sin"]),
                    attention_mask=causal_mask_mapping[layer_type],
                    position_ids=position_ids,
                    past_key_values=self.past_key_values
                )

            # Warmup first token
            self.past_key_values.commit_tokens = 0

            causal_mask_mapping = self.create_attention_masks(
                input_embeddings=hidden_states[:, 0:1, :],
                attention_mask=attention_mask,
                token_position=0,
                past_key_values=None,
                position_ids=position_ids,
                mask_layer_indices=self.upper_layers_attention
            )
            warmup_state = self.warmup_pass(
                hidden_states=hidden_states[:, 0:1, :].clone(),
                position_ids=position_ids[:, 0:1],
                position_embeddings=self._slice_position_embeddings(position_embeddings, start=0, end=1),
                causal_mask_mapping=causal_mask_mapping
            )

            self.final_hidden_states.append(warmup_state)            

            for token_position in range(hidden_states.shape[1] - 1):
                # Recirculate first K tokens and warmup the next K tokens
                self.past_key_values.commit_tokens = K
                causal_mask_mapping = self.create_attention_masks(
                    input_embeddings=hidden_states[:, token_position:token_position+2*K, :],
                    attention_mask=attention_mask,
                    token_position=token_position,
                    past_key_values=self.past_key_values,
                    position_ids=position_ids,
                    mask_layer_indices=self.upper_layers_attention
                )

                hidden_states_block = self.recirculated_block_pass(
                    hidden_states=hidden_states[:, token_position+K:token_position+2*K, :],
                    position_ids=position_ids[:, token_position:token_position+2*K],
                    position_embeddings=self._slice_position_embeddings(position_embeddings, start=token_position, end=token_position+2*K),
                    causal_mask_mapping=causal_mask_mapping,
                    K=K,
                    alpha=self._compute_effective_alpha(token_position)
                )

                self.final_hidden_states.append(hidden_states_block[:, K:2*K, :])

            final_hidden_states = torch.cat(self.final_hidden_states, dim=1)
            final_hidden_states = self.model_decoder.norm(final_hidden_states)
            logits = self.model.lm_head(final_hidden_states)

            return logits, input_ids

    def _slice_position_embeddings(self, position_embeddings, start, end):
        sliced_position_embeddings = {}
        for key in list(position_embeddings.keys()):
            sliced_position_embeddings[key] = {
                "cos": position_embeddings[key]["cos"][:, start:end],
                "sin": position_embeddings[key]["sin"][:, start:end],
            }

        return sliced_position_embeddings

    def warmup_pass(
            self,
            hidden_states,
            position_ids,
            position_embeddings,
            causal_mask_mapping
        ):
        """Warmup pass for first token of a sequence."""

        for layer in range(self.destination + 1, self.model_config.num_hidden_layers):
            if layer == self.destination + 1:
                self.destination_hidden = hidden_states.clone()

            if layer == self.source + 1:
                self.source_hidden = hidden_states.clone()
            
            layer_type = self.model_config.layer_types[layer]
    
            hidden_states = self.model_decoder.layers[layer].forward(
                hidden_states=hidden_states,
                position_embeddings=(position_embeddings[layer_type]["cos"], position_embeddings[layer_type]["sin"]),
                attention_mask=causal_mask_mapping[layer_type],
                position_ids=position_ids,
                past_key_values=self.past_key_values
            )

        return hidden_states

    def recirculated_block_pass(
            self,
            hidden_states,
            position_ids,
            position_embeddings,
            causal_mask_mapping,
            K,
            alpha
        ):
        """Recirculated pass for block size K and normal pass for the subsequent block size K.""" 
        source_scaled = self.source_hidden  * (self.destination_hidden.norm(dim=-1, keepdim=True) / self.source_hidden.norm(dim=-1, keepdim=True))

        recirculated_token = alpha * source_scaled + self.beta * self.destination_hidden

        hidden_states = torch.cat([recirculated_token, hidden_states], dim=1)

        for layer in range(self.destination + 1, self.model_config.num_hidden_layers):
            if layer == self.destination + 1:
                self.destination_hidden = hidden_states[:, K:2*K, :].clone()
            
            if layer == self.source + 1:
                self.source_hidden = hidden_states[:, K:2*K, :].clone()

            layer_type = self.model_config.layer_types[layer]

            hidden_states = self.model_decoder.layers[layer].forward(
                hidden_states=hidden_states,
                position_embeddings=(position_embeddings[layer_type]["cos"], position_embeddings[layer_type]["sin"]),
                attention_mask=causal_mask_mapping[layer_type],
                position_ids=position_ids,
                past_key_values=self.past_key_values
            )

        return hidden_states

    def recirculate_token(
            self,
            position_ids,
            position_embeddings,
            causal_mask_mapping
            ):
        """Recirculates hidden state from source layer to destinaton layer for first token in iteration."""
        source_scaled = self.source_hidden  * (self.destination_hidden.norm(dim=-1, keepdim=True) / self.source_hidden.norm(dim=-1, keepdim=True))

        hidden_states = self.alpha * source_scaled + self.beta * self.destination_hidden

        for layer in range(self.destination + 1, self.model_config.num_hidden_layers):
            layer_type = self.model_config.layer_types[layer]

            hidden_states = self.model_decoder.layers[layer].forward(
                hidden_states=hidden_states,
                position_embeddings=(position_embeddings[layer_type]["cos"], position_embeddings[layer_type]["sin"]),
                attention_mask=causal_mask_mapping[layer_type],
                position_ids=position_ids,
                past_key_values=self.past_key_values
            )

        return hidden_states


    def generate_input_embeddigs(self, input_ids):
        """Generates token input embeddings."""
        ## Token Embeddings
        return self.model_decoder.embed_tokens.forward(input_ids=input_ids)

    def generate_position_embeddings(self, hidden_states, start=0, end=None):
        """Generate token positional embeddings."""
        ## Position Embeddings
        if end is None:
            position_ids = torch.arange(
                start=start,
                end=start + hidden_states.shape[1],
                device=hidden_states.device
            ).unsqueeze(0)
        else:
            position_ids = torch.arange(
                start=start,
                end=end,
                device=hidden_states.device
            ).unsqueeze(0)

        position_embeddings = {}

        for layer_type in set(self.model_config.layer_types):
            cos, sin = self.model_decoder.rotary_emb(
                hidden_states,
                position_ids,
                layer_type
            )

            position_embeddings[layer_type] = {
                "cos": cos,
                "sin": sin
            }

        return position_ids, position_embeddings

    def recirculated_generate(
            self,
            input_ids,
            attention_mask,
            max_new_tokens=10,
            K=1
        ):
        generated_logits = []
        generated_ids = []

        with torch.no_grad():
            ## Prefill and first token prediction

            ## Empty Cache
            self.past_key_values = RecirculationDynamicCache(config=self.model_config)

            ## Generate Embeddings
            input_embeddings = self.generate_input_embeddigs(input_ids=input_ids)
            position_ids, position_embeddings = self.generate_position_embeddings(hidden_states=input_embeddings)

            ## Prefill & First-Next-Token Generation
            logits, _ = self.recirculated_run(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_embeddings=input_embeddings,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                reset_cache=False,
                K=K
            )
        
            next_logits = logits[:, -1, :]
            next_token_id = next_logits.argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
            ], dim=1)


            generated_logits.append(next_logits)
            generated_ids.append(next_token_id)

            ## Iterative next token prediction
            for _ in range(max_new_tokens-1):
                # Write an iterative method for autoregressive_prediction `autoregressive_run`
                input_sequence_len = input_ids.shape[1]

                # Generate embeddings
                hidden_states = self.generate_input_embeddigs(next_token_id)
                position_ids, position_embeddings = self.generate_position_embeddings(hidden_states=hidden_states, start=input_sequence_len-K-1, end=input_sequence_len+K-1)

                # Lower layer pass for new token
                self.past_key_values.commit_tokens = None

                causal_mask_mapping = self.create_attention_masks(
                    input_embeddings=hidden_states,
                    attention_mask=attention_mask,
                    token_position=None,
                    past_key_values=self.past_key_values,
                    position_ids=position_ids[:,-K:],
                    mask_layer_indices=self.lower_layers_attention
                )

                for layer in range(self.destination + 1):
                    layer_type = self.model_config.layer_types[layer]

                    sliced_position_embeddings = self._slice_position_embeddings(position_embeddings, start=K, end=2*K)
                        
                    hidden_states = self.model_decoder.layers[layer].forward(
                        hidden_states=hidden_states,
                        position_embeddings=(sliced_position_embeddings[layer_type]["cos"], sliced_position_embeddings[layer_type]["sin"]),
                        attention_mask=causal_mask_mapping[layer_type],
                        position_ids=position_ids[:,-K:],
                        past_key_values=self.past_key_values
                    )

                # Recirculate K tokens
                self.past_key_values.commit_tokens = K

                causal_mask_mapping = self.create_attention_masks(
                    input_embeddings=torch.cat([hidden_states, hidden_states], dim=1),  # dummy input_embeddings for mask creation
                    attention_mask=attention_mask,
                    token_position=None,
                    past_key_values=self.past_key_values,
                    position_ids=position_ids,
                    mask_layer_indices=self.upper_layers_attention
                )

                hidden_states = self.recirculated_block_pass(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    causal_mask_mapping=causal_mask_mapping,
                    K=K,
                    alpha=self._compute_effective_alpha(token_position = input_sequence_len - K - 1)
                )

                # Accumulate final hidden states
                final_hidden_state = hidden_states[:, K:2*K, :]
                self.final_hidden_states.append(final_hidden_state)

                final_hidden_state = self.model_decoder.norm(final_hidden_state)
                next_logits = self.model.lm_head(final_hidden_state)
                next_logits = next_logits[:, -1, :]
                next_token_id = next_logits.argmax(dim=-1, keepdim=True)
    
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                ], dim=1)

                generated_logits.append(next_logits)
                generated_ids.append(next_token_id)

                
            generated_logits = torch.stack(generated_logits, dim=1)
            generated_ids = torch.cat(generated_ids, dim=1)

            return generated_logits, generated_ids



    def create_attention_masks(
            self,
            input_embeddings,
            attention_mask,
            token_position,
            past_key_values,
            position_ids,
            mask_layer_indices
        ):
        if token_position is not None:
            num_of_tokens = input_embeddings.shape[1]
            end = num_of_tokens + token_position

            attention_mask = attention_mask[:, :end]
            position_ids = position_ids[:, token_position:end]
        
        mask_kwargs = {
            "config": self.model_config,
            "inputs_embeds": input_embeddings,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        
        # Create the masks
        causal_mask_mapping = {}

        if "full_attention" in mask_layer_indices.keys():
            causal_mask_mapping["full_attention"] = create_causal_mask(**mask_kwargs, layer_idx=mask_layer_indices["full_attention"])
        if "sliding_attention" in mask_layer_indices.keys():
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs, layer_idx=mask_layer_indices["sliding_attention"])

        return causal_mask_mapping

    def vanilla_run(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                        use_cache=False,
                    )

        return outputs.logits, input_ids
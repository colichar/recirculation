import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.masking_utils import (create_causal_mask, create_sliding_window_causal_mask)
from recirculated_cache import RecirculationDynamicCache


class RecirculatedGemma:
    def __init__(self, source=11, destination=4, alpha=0.25, model_name="google/gemma-3-1b-it", dtype: torch.dtype = torch.float32):
        self.source = source
        self.destination = destination
        self.alpha = alpha

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained( 
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

        self.past_key_values = None
        self.destination_hidden = None
        self.source_hidden = None
        self.position_ids = None
        self.causal_mask_mapping = None
        self.position_embeddings = {}

        self.final_hidden_states = []

        upper_layers = range(
            self.destination + 1,
            self.model.config.num_hidden_layers
        )

        self.upper_full_layers = next(
            i for i in upper_layers
            if self.model.config.layer_types[i] == "full_attention"
        )

        self.upper_sliding_layers = next(
            i for i in upper_layers
            if self.model.config.layer_types[i] == "sliding_attention"
        )

    def run(self, input_ids, attention_mask=None):
        with torch.no_grad():
            ## Empty Cache
 
            self.past_key_values = RecirculationDynamicCache(config=self.model.config)
            self.final_hidden_states = []

            ## Token Embeddings

            input_embeddings = self.model.model.embed_tokens.forward(input_ids=input_ids)

            hidden_states = input_embeddings.clone()
            ## Position Embeddings

            self.position_ids = torch.arange(
                hidden_states.shape[1],
                device=hidden_states.device
            ).unsqueeze(0)

            for layer_type in set(self.model.config.layer_types):
                cos, sin = self.model.model.rotary_emb(
                    hidden_states,
                    self.position_ids,
                    layer_type
                )

                self.position_embeddings[layer_type] = {
                    "cos": cos,
                    "sin": sin
                }

            # Warmup all tokens
            self.past_key_values.commit_tokens = None

            self.create_attention_masks(
                token_embedding=hidden_states,
                attention_mask=attention_mask,
                token_position=None,
                past_key_values=self.past_key_values
            )

            for layer in range(self.destination + 1):
                layer_type = self.model.config.layer_types[layer]
                    
                hidden_states = self.model.model.layers[layer].forward(
                    hidden_states=hidden_states,
                    position_embeddings=(self.position_embeddings[layer_type]["cos"], self.position_embeddings[layer_type]["sin"]),
                    attention_mask=self.causal_mask_mapping[layer_type],
                    position_ids=self.position_ids,
                    past_key_values=self.past_key_values
                )

            # Warmup first token
            self.past_key_values.commit_tokens = 0

            self.create_attention_masks(
                token_embedding=hidden_states[:, 0:1, :],
                attention_mask=attention_mask,
                token_position=0,
                past_key_values=None
            )
            self.warmup_pass(hidden_states=hidden_states[:, 0:1, :].clone())

            # Currently hardcoded block-size K
            K = 1

            for token_position in range(hidden_states.shape[1] - 1):
                # Recirculate first K tokens and warmup the next K tokens
                self.past_key_values.commit_tokens = K
                self.create_attention_masks(
                    token_embedding=hidden_states[:, token_position:token_position+2*K, :],
                    attention_mask=attention_mask,
                    token_position=token_position,
                    past_key_values=self.past_key_values
                )

                hidden_states_block = self.recirculated_block_pass(
                    hidden_states=hidden_states[:, token_position:token_position+2*K, :],
                    token_position=token_position
                )

                # Accumulate final hidden states
                self.final_hidden_states.append(hidden_states_block[:, :K, :])


            final_token = input_embeddings.shape[1] - 1
            self.create_attention_masks(
                token_embedding=input_embeddings[:, final_token:final_token+1, :],
                attention_mask=attention_mask,
                token_position=final_token,
                past_key_values=self.past_key_values
            )

            self.past_key_values.commit_tokens = 0
            final_state = self.recirculate_token(token_position=final_token)

            self.final_hidden_states.append(final_state)

            final_hidden_states = torch.cat(self.final_hidden_states, dim=1)
            final_hidden_states = self.model.model.norm(final_hidden_states)
            logits = self.model.lm_head(final_hidden_states)

            return logits, input_ids

    def warmup_pass(self, hidden_states):
        """Warmup pass for first token of a sequence."""

        for layer in range(self.destination + 1, self.model.config.num_hidden_layers):
            if layer == self.destination + 1:
                self.destination_hidden = hidden_states.clone()

            if layer == self.source + 1:
                self.source_hidden = hidden_states.clone()
            
            layer_type = self.model.config.layer_types[layer]
    
            hidden_states = self.model.model.layers[layer].forward(
                hidden_states=hidden_states,
                position_embeddings=(self.position_embeddings[layer_type]["cos"][:, 0:1], self.position_embeddings[layer_type]["sin"][:, 0:1]),
                attention_mask=self.causal_mask_mapping[layer_type],
                position_ids=self.position_ids[:, 0:1],
                past_key_values=self.past_key_values
            )

        return hidden_states

    def recirculated_block_pass(self, hidden_states, token_position, K=1):
        """Recirculated pass for block size K and normal pass for the subsequent block size K.""" 
        source_scaled = self.source_hidden  * (self.destination_hidden.norm(dim=-1, keepdim=True) / self.source_hidden.norm(dim=-1, keepdim=True))

        hidden_states[:, 0:K, :] = self.alpha * source_scaled + (1 - self.alpha) * self.destination_hidden

        for layer in range(self.destination + 1, self.model.config.num_hidden_layers):
            if layer == self.destination + 1:
                self.destination_hidden = hidden_states[:, K:2*K, :].clone()
            
            if layer == self.source + 1:
                self.source_hidden = hidden_states[:, K:2*K, :].clone()

            layer_type = self.model.config.layer_types[layer]

            hidden_states = self.model.model.layers[layer].forward(
                hidden_states=hidden_states,
                position_embeddings=(self.position_embeddings[layer_type]["cos"][:, token_position:token_position+2*K], self.position_embeddings[layer_type]["sin"][:, token_position:token_position+2*K]),
                attention_mask=self.causal_mask_mapping[layer_type],
                position_ids=self.position_ids[:, token_position:token_position+2*K],
                past_key_values=self.past_key_values
            )

        return hidden_states
        

    def recirculate_token(self, token_position):
        """Recirculates hidden state from source layer to destinaton layer for first token in iteration."""
        source_scaled = self.source_hidden  * (self.destination_hidden.norm(dim=-1, keepdim=True) / self.source_hidden.norm(dim=-1, keepdim=True))

        hidden_states = self.alpha * source_scaled + (1 - self.alpha) * self.destination_hidden

        for layer in range(self.destination + 1, self.model.config.num_hidden_layers):
            layer_type = self.model.config.layer_types[layer]

            hidden_states = self.model.model.layers[layer].forward(
                hidden_states=hidden_states,
                position_embeddings=(self.position_embeddings[layer_type]["cos"][:, token_position:token_position+1], self.position_embeddings[layer_type]["sin"][:, token_position:token_position+1]),
                attention_mask=self.causal_mask_mapping[layer_type],
                position_ids=self.position_ids[:, token_position:token_position+1],
                past_key_values=self.past_key_values
            )

        return hidden_states

    def generate(self, input_ids, attention_mask, max_new_tokens=10):
        generated_logits = []
        generated_ids = []


        for _ in range(max_new_tokens):
            logits, _ = self.run(input_ids=input_ids, attention_mask=attention_mask)

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

        generated_logits = torch.stack(generated_logits, dim=1)
        generated_ids = torch.cat(generated_ids, dim=1)

        return generated_logits, generated_ids

    def create_attention_masks(self, token_embedding, attention_mask, token_position, past_key_values):
        if token_position is not None:
            num_of_tokens = token_embedding.shape[1]
            end = num_of_tokens + token_position

            attention_mask = attention_mask[:, :end]
            position_ids = self.position_ids[:, token_position:end]
        else:
            position_ids = self.position_ids
        
        mask_kwargs = {
            "config": self.model.config,
            "inputs_embeds": token_embedding,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        sliding_mask_kwargs = mask_kwargs.copy()
        
        # Create the masks
        self.causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs, layer_idx=self.upper_full_layers),
            "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs, layer_idx=self.upper_sliding_layers),
        }

    def run_vanilla(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                        use_cache=False,
                    )

        return outputs.logits, input_ids

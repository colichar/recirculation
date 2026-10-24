# Numerical integration tests

The Gemma integration tests use deterministic, randomly initialized tiny models with both the 1B-style `Gemma3ForCausalLM` layout and the 4B-style `Gemma3ForConditionalGeneration` layout. They check execution and cache behavior—not full-checkpoint accuracy or paper results.

The checks run in CPU FP32, CUDA FP32, and CUDA BF16, when the hardware supports them. CUDA tests skip explicitly when CUDA or native BF16 support is unavailable.

- **Zero-alpha equivalence:** With `alpha=0`, compare `recirculated_run` logits with vanilla model logits. Sequence lengths include cases beyond the sliding window.
- **Causality:** With nonzero alpha, change later input tokens and check that earlier `recirculated_run` logits remain unchanged.
- **First-pass readout:** Compare `recirculated_run` logits with the first-pass hidden states captured from the warmup and paired passes.
- **Execution dtype and device:** Check that model parameters and KV cache tensors use the requested device and dtype.
- **Sequential reference:** Compare logits and cached keys/values with an independent token-by-token Recirculation implementation.
- **Padding:** Verify that right padding preserves real-token outputs and that attention excludes masked positions.

Logits are compared in FP32, including outputs from BF16 execution. See `tests/test_recirculated_gemma_cache.py` for the test cases and tolerances.

Run the tests with:

```bash
uv run --extra cuda python -m pytest tests -q
```
Use --extra cpu on a CPU-only machine.


## Tolerances

All logit comparisons and difference measurements are performed in FP32, allowing for a vanilla implementation that upcasts its logits. Casting BF16 logits to FP32 for comparison does not change the dtype used for model execution.

| Comparison | Execution precision | `rtol` | `atol` |
| --- | --- | --- | --- |
| Zero-alpha vs. vanilla | CPU/CUDA FP32 | `1e-5` | `1e-6` |
| Zero-alpha vs. vanilla | CUDA BF16 | `1e-2` | `1e-5` |
| Causality and first-pass readout | All three configurations | `1e-5` | `1e-6` |
| Sequential-reference logits | CPU/CUDA FP32 | `1e-5` | `1e-6` |
| Sequential-reference KV | CPU/CUDA FP32 | `1e-5` | `3e-6` |
| Sequential-reference logits/KV | CUDA BF16 | `1e-2` | `1e-5` |

- Zero-alpha equivalence uses `rtol=1e-5`, `atol=1e-6` for CPU/CUDA FP32 and `rtol=1e-2`, `atol=1e-5` for CUDA BF16. The BF16 tolerance allows for rounding differences between vanilla and paired-pass execution.
- Causality and first-pass readout checks use `rtol=1e-5`, `atol=1e-6` across all configurations because they compare matching computations.
- Sequential-reference KV checks allow small rounding differences from the different execution order (1.49e−6 observed; 3e−6 absolute tolerance). These checks have passed on CPU FP32 and CUDA FP32/BF16.

These tolerances apply only to the tiny, randomly initialized integration models. Manual comparisons with google/gemma-3-4b-pt exceeded them; no separate numerical acceptance threshold has been set for the pretrained checkpoint. Unexpected test failures should be investigated before increasing tolerances.
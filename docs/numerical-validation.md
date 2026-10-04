# Numerical integration tests

The Gemma integration tests use a deterministic, randomly initialized tiny model. They check execution and cache behavior—not full-checkpoint accuracy or paper results.

The checks run in CPU FP32, CUDA FP32, and CUDA BF16, when the hardware supports them. CUDA tests skip explicitly when CUDA or native BF16 support is unavailable.

- **Zero-alpha equivalence:** With `alpha=0`, compare Recirculation logits with vanilla model logits in both `first_pass` and `recirculated_pass` modes. Sequence lengths include cases beyond the sliding window.
- **Causality:** With nonzero alpha, change later input tokens and check that earlier logits remain unchanged, in both readout modes.
- **Readout selection:** Compare each readout with the hidden states captured from its corresponding model pass. Check that the default selects `first_pass`.
- **Execution dtype and device:** Check that model parameters and KV cache tensors use the requested device and dtype.

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
| Causality and output selection | All three configurations | `1e-5` | `1e-6` |

- Zero-alpha equivalence uses `rtol=1e-5`, `atol=1e-6` for CPU/CUDA FP32 and `rtol=1e-2`, `atol=1e-5` for CUDA BF16. The BF16 tolerance allows for rounding differences between vanilla and paired-pass execution.
- Causality and readout-selection checks use `rtol=1e-5`, `atol=1e-6` across all configurations because they compare matching computations.

These tolerances apply to the tiny integration model. Unexpected failures should be investigated before increasing them.
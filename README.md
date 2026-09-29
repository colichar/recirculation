# Recirculation

This project implements DeepMind's **Recirculation** method for Transformer language models and investigates its behavior in a reproducible experimental setting. The goal is to reproduce key results from the original paper, evaluate the method through controlled ablations, and explore extensions and modifications to the original approach.

## Environment setup

Requirements:

- Python 3.12 (uv downloads the pinned version if needed)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

To install the dependencies, run one of the following commands from the repository root:

```bash
uv sync --extra cpu     # CPU-only machine
uv sync --extra cuda    # NVIDIA GPU with a CUDA 12.6-compatible driver
```

**Note:** Always pass `--extra cpu` or `--extra cuda` to `uv sync` and `uv run`.
Otherwise, Accelerate's dependency on PyTorch can select a different build.

For VSCode notebooks, select `.venv/bin/python` as the kernel.

## Running tests

From the repository root:

```bash
uv run --extra cpu python -m pytest tests -q
```

Use `--extra cuda` instead if that is your installed backend.

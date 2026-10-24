# AGENTS.md

## Project Context

This is a research project, not a production AI engineering project.

The goal is to reproduce DeepMind's Recirculation method, investigate its behavior through controlled experiments and ablations, explore extensions to the method, and summarize the findings in a research report.

Scientific correctness, experimental clarity, and iteration speed take priority over production-grade software engineering.

## Core Principles

- Prefer the simplest solution that correctly supports the current experiment.
- Keep code clear, explicit, and easy for a human to read and verify.
- Avoid unnecessary abstractions, frameworks, configuration layers, and architectural complexity.
- Do not optimize for hypothetical future requirements.
- Do not generalize an implementation unless the project currently requires it.
- Prefer research iteration speed over production robustness when the two conflict.
- Preserve scientific correctness and reproducibility when simplifying code.
- **Paper fidelity comes first when implementing the original method. Clearly distinguish paper-faithful Recirculation from implementation optimizations and experimental variants.**
- **Prefer a simple/reference implementation for establishing correctness before optimizing it.**
- **Never silently introduce changes that could affect model behavior, logits, cache semantics, evaluation results, or scientific conclusions. Make such changes explicit.**

## Scope of AI-Generated Code

By default, AI may independently implement:

- Tests.
- Test fixtures and utilities.
- Boilerplate.
- Straightforward supporting code.
- Mechanical refactors that do not change behavior.

Inference, recirculation, evaluation, and experimental logic are primarily implemented by the project author.

Do not implement or substantially modify this core logic unless explicitly asked to do so. AI may review, explain, debug, or propose changes to it without taking over its implementation.

## Implementation Guidelines

Write code for the scenarios that are realistically expected in this project.

In particular:

- Add branches only for cases that can realistically occur.
- Add guards only for invalid states that can realistically be reached.
- Handle errors that are relevant to the actual experiments.
- Do not add defensive code for hypothetical misuse.
- Do not introduce extensibility solely because it might be useful later.
- **Avoid adding dependencies when existing dependencies or a small, clear implementation are sufficient.**

If a simpler implementation works for the current experimental scope, prefer it over a more general implementation.

## Project Structure

- `recirculation/` — Recirculation implementation and supporting model/cache code.
- `tests/` — Tests for cache, Recirculation, and model behavior.
- `notebooks/` — Exploratory analysis, debugging, and experiments.
- `evaluation/` — Evaluation and paper-reproduction code.

## Testing

Tests should protect scientifically or technically meaningful behavior.

- Test behavior and invariants that matter for the experiments.
- Prioritize tests for numerical correctness, cache semantics, masking, recirculation behavior, evaluation logic, and other assumptions that could invalidate experimental results.
- Include boundary cases when they can realistically occur during the planned experiments.
- Do not create tests solely to increase coverage.
- Do not exhaustively test impossible or irrelevant inputs.
- Prefer a small number of meaningful tests over a large defensive test suite.
- **Use appropriate numerical tolerances when comparing model outputs. Do not assume exact equality where floating-point precision or execution order can reasonably cause differences, particularly with BF16.**

A test should ideally answer: **what realistic failure would this test protect the research from?**

## Experimental Guidelines

- **Keep baselines, Recirculation, and ablations under equivalent evaluation conditions wherever possible. Change only the variables required by the experiment.**
- **Record enough information to trace important experiments, including the model/checkpoint, dataset, relevant hyperparameters, Recirculation configuration, and relevant code/configuration state.**
- **Do not automatically treat disagreement with the paper or an unexpected result as an implementation bug. Verify the implementation and experimental setup, then treat unexplained discrepancies as results to investigate.**
- **Keep paper reproduction results clearly separated from experimental modifications and extensions.**

## Code Style

Code should be:

- Simple.
- Explicit.
- Human-readable.
- Easy to inspect and debug.
- Consistent with the existing codebase.

Prefer straightforward code over clever or highly abstract solutions. Optimize for a researcher being able to understand the implementation and relate it to the method being studied.

## Documentation

- Keep documentation lean, factual, and human-readable.
- Focus on what is implemented, checked, or required; explain why only when needed to avoid misunderstanding.
- Do not duplicate code, tables, or information already stated elsewhere.
- For numerical validation, record coverage, tolerances, run commands, and verification status.
- Keep tolerance rationale brief; omit debugging narratives and incidental measurements unless needed to justify a numerical threshold.

### Docstrings

- Start with one sentence stating the purpose; use only that sentence for simple helpers.
- Document behavior and assumptions needed for correct use, not line-by-line implementation details.
- Use Args, Returns (or Yields), and Raises sections when useful; omit unnecessary sections.
- For tensor inputs and outputs, explain their shape and meaning where needed for correct use.
- Describe return structure and meaningful side effects, especially cache updates or resets.
- For classes, summarize their responsibility and important state; avoid duplicating method documentation.
- Keep parameter names and documented behavior consistent with the code.

## Reproducibility and Users

The project is currently optimized primarily for research by its author.

Reproducibility by other users is desirable, but secondary to reaching the project's experimental goals efficiently. Do not introduce substantial complexity solely to improve external usability at this stage.

As the project approaches completion, documentation, reproducibility, configuration, and usability for other researchers can be strengthened.

## Decision Rule

When choosing between implementations, prefer the option that:

1. Preserves scientific correctness.
2. Makes the experiment easier to understand and verify.
3. Uses the least unnecessary complexity.
4. Enables faster research iteration.

Production robustness, broad API design, and support for hypothetical users are secondary concerns unless explicitly requested.
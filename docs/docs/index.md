# AumOS Tabular Engine

Enterprise-grade synthetic tabular data generation with built-in differential privacy,
multi-table support, and quality gating.

## What it does

The tabular engine generates high-fidelity synthetic datasets from your real tabular data:

- **Statistical fidelity** — CTGAN, TVAE, and GaussianCopula generators preserve distributions, correlations, and relationships
- **Differential privacy** — SmartNoise integration with formal epsilon/delta budget tracking
- **Multi-table synthesis** — HMA generator preserves foreign key cardinalities across table hierarchies
- **Quality gates** — SDMetrics fidelity scoring and Anonymeter privacy risk assessment before release
- **Time-series** — PAR and DeepEcho generators for sequential/temporal tabular data

## Architecture

```
source data (MinIO) → schema analysis → generator training → synthesis → quality gate → output (MinIO)
                                             ↕
                                   aumos-privacy-engine (epsilon budget)
```

## Quickstart

See [Quickstart](quickstart.md) to generate your first synthetic dataset in 5 minutes.

## Supported Generators

| Generator | Type | DP Support | Best For |
|-----------|------|-----------|---------|
| `ctgan` | Deep learning | No | Complex distributions, correlations |
| `tvae` | Deep learning | No | Structured tabular data |
| `gaussian_copula` | Statistical | No | Fast generation, continuous data |
| `smartnoise` | DP-enabled | Yes | Regulated industries, PII-adjacent data |
| `hma` | Multi-table | No | Hierarchical schemas with FK relationships |
| `par` | Time-series | No | Sequential/temporal tabular data |
| `deepecho` | Time-series | No | Long-dependency sequences |

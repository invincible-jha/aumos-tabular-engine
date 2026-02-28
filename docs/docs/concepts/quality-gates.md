# Quality Gates

Every generation job passes through an automated quality gate before the output is released.

## Metrics

### Fidelity Score (SDMetrics)

Measures how statistically similar the synthetic data is to the real data:
- **Column Shapes** — distribution matching per column (KS test for numeric, TVComplement for categorical)
- **Column Pair Trends** — correlation preservation between column pairs
- **Overall Score** — weighted average across all columns and pairs

Default minimum: `0.75` (configurable via `AUMOS_MIN_FIDELITY_SCORE`)

### Privacy Score (Anonymeter)

Measures re-identification risk:
- **Singling Out** — can any synthetic record uniquely identify a real individual?
- **Linkability** — can combinations of quasi-identifiers link synthetic to real records?

Default minimum: `0.50` (configurable via `AUMOS_MIN_PRIVACY_SCORE`)

## Gate Failure Behavior

If either score falls below threshold, the job transitions to `FAILED` with:
- `error_message` explaining which gate failed and the actual score
- All intermediate artifacts preserved for debugging
- Job is not retried automatically — fix the configuration and resubmit

## Bypassing Gates (Dev Only)

Set `AUMOS_ENV=development` and `quality_threshold=0.0` to skip quality gating during development.
Never bypass gates in production.

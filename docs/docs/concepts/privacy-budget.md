# Privacy Budget

The tabular engine integrates with `aumos-privacy-engine` for formal epsilon/delta budget tracking.

## How It Works

Every `smartnoise` generation job:
1. Requests epsilon allocation from the privacy engine before training
2. If the tenant has insufficient remaining budget, the job fails immediately with a clear error
3. After generation completes, the allocated budget is consumed and recorded

## Epsilon Budget Protocol

```
POST /api/v1/generation/jobs  (model=smartnoise, dp_epsilon=0.5)
    ↓
aumos-privacy-engine: allocate_budget(tenant_id, epsilon=0.5)
    ↓ (if approved)
SmartNoise training + generation
    ↓
aumos-privacy-engine: consume_budget(allocation_id)
    ↓
Job → COMPLETE
```

## Time-Series DP

For time-series jobs (`par` or `deepecho` generators with `dp_mode=true`), epsilon is
allocated **per sequence** rather than per row. This is the correct privacy unit —
an attacker tries to identify a person's sequence, not an individual row within a sequence.

## Choosing Epsilon

| Epsilon | Privacy Level | Data Utility |
|---------|---------------|--------------|
| 0.1 | Very high | Lower fidelity |
| 1.0 | High (recommended) | Good fidelity |
| 5.0 | Moderate | High fidelity |
| 10.0 | Low | Near-original |

## Budget Monitoring

Check remaining budget:

```bash
curl http://localhost:8006/api/v1/privacy/budget \
  -H "Authorization: Bearer $API_KEY"
```

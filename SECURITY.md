# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security vulnerabilities to the AumOS security team:

**Email:** security@aumos.io

### What to Include

- Description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept if possible)
- Affected versions
- Any suggested mitigations

### Response Timeline

- **Acknowledgement:** Within 48 hours
- **Initial assessment:** Within 5 business days
- **Fix + coordinated disclosure:** Within 90 days for critical issues

We follow responsible disclosure principles and will coordinate with you before
publishing any details publicly.

## Security Considerations for tabular-engine

### Differential Privacy

- Always verify epsilon/delta budgets before generation jobs
- The privacy engine enforces per-tenant budget caps
- Never bypass the `aumos-privacy-engine` budget check for DP jobs

### Data Handling

- Input datasets containing PII must be pre-approved by your data governance team
- Generated synthetic data does not inherit the classification of source data
- Audit logs for all generation jobs are retained per your tenant's retention policy

### Multi-Tenancy

- Row-level security (RLS) is enforced at the database layer
- Never expose tenant IDs from one tenant's context to another
- All API endpoints require a valid JWT with the correct tenant claim

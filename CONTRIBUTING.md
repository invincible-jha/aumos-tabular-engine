# Contributing to aumos-tabular-engine

Thank you for your interest in contributing to the AumOS Tabular Engine.

## License Restrictions

**Important:** Contributions incorporating AGPL, GPL, or LGPL licensed code are
not accepted. All dependencies must be permissively licensed (Apache-2.0, MIT, BSD).

## Getting Started

1. Fork the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run the test suite before making changes:
   ```bash
   make test
   ```

## Development Workflow

### Code Style

- Python 3.11+ with strict type hints on all function signatures
- Line length: 120 characters (enforced by ruff)
- Use `ruff` for linting and formatting
- Use `mypy` in strict mode for type checking

```bash
make format    # auto-format with ruff
make lint      # lint check
make typecheck # mypy strict
```

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add SmartNoise DP generator with Laplace mechanism
fix: correct epsilon budget allocation in privacy client
refactor: extract constraint validation into dedicated engine
docs: document differential privacy parameter recommendations
test: add integration tests for CTGAN generator
chore: upgrade sdv to 1.11.0
```

Commit messages must explain **why**, not just what.

### Testing

- Write tests alongside new code, not as an afterthought
- Unit tests in `tests/unit/`, integration tests in `tests/integration/`
- Target 80% coverage minimum
- Mock external services (Kafka, MinIO, privacy engine) in unit tests

```bash
make test        # full suite with coverage
make test-quick  # fast subset, fail fast
```

### Pull Requests

1. Ensure `make all` passes (lint + typecheck + test)
2. Keep PRs focused — one logical change per PR
3. Reference related issues in the PR description
4. PRs are squash-merged to keep history clean

## Architecture

This service follows hexagonal architecture:

- `core/` — domain models, interfaces, business logic (no framework dependencies)
- `api/` — FastAPI routers, Pydantic schemas, dependency injection
- `adapters/` — concrete implementations (SQLAlchemy, generators, Kafka, MinIO)

Never import from `adapters/` in `core/`. Use `core/interfaces.py` Protocol classes.

## Community Governance

This project follows the AumOS Community Governance model.
See [aumos-community-governance](https://github.com/aumos-enterprise/aumos-community-governance)
for code of conduct, decision-making process, and escalation paths.

## Security

Report vulnerabilities to security@aumos.io — see SECURITY.md for full details.
Never open public GitHub issues for security vulnerabilities.

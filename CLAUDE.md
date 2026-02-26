# aumos-tabular-engine — CLAUDE.md

## Repo Purpose

Enterprise-grade tabular synthetic data generation service. Generates high-fidelity
synthetic datasets from real tabular data using CTGAN, TVAE, GaussianCopula
(statistical) and SmartNoise (differential privacy) generators.

Core responsibilities:
1. **Schema analysis** — auto-detect column types, distributions, constraints
2. **Synthetic generation** — train generative models and produce synthetic rows
3. **DP enforcement** — call `aumos-privacy-engine` to allocate and track epsilon budgets
4. **Quality validation** — SDMetrics fidelity + Anonymeter privacy risk scoring
5. **Multi-table synthesis** — FK-preserving hierarchical multi-table generation
6. **Artifact storage** — MinIO-backed dataset storage with presigned download URLs

## Architecture Position

```
Upstream (depends on):
  aumos-common        → base models, errors, health, config, observability
  aumos-proto         → Protobuf event schemas for Kafka
  aumos-privacy-engine → DP budget allocation (HTTP)
  aumos-model-registry → generator model versioning (HTTP)

Downstream (provides to):
  aumos-fidelity-validator → sends generated datasets for deeper validation
  aumos-data-pipeline      → synthesis is one step in orchestrated pipelines
  aumos-marketplace        → exposes synthetic datasets for discovery
  aumos-healthcare-synth   → domain-specific extension
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Web framework | FastAPI 0.110+ |
| Validation | Pydantic v2 (strict mode) |
| Config | pydantic-settings with `AUMOS_` prefix |
| Database | PostgreSQL 16 via SQLAlchemy 2.0 async + asyncpg |
| Statistical generators | SDV 1.10+ (GaussianCopula, TVAE, HMA) |
| Deep learning generator | CTGAN 0.9+ |
| DP generator | SmartNoise-Synth 1.0+ |
| Quality evaluation | SDMetrics 0.14+ |
| Privacy risk | Anonymeter 1.0+ |
| Data validation | Great Expectations 0.18+ |
| DataFrames | pandas 2.2+, polars 0.20+, pyarrow 15+ |
| Artifact storage | MinIO via minio SDK |
| Messaging | Kafka via aiokafka |
| Observability | structlog via aumos_common.observability |
| Containerisation | Docker multi-stage, non-root user |

## Database Table Prefix

**All tables use the `tab_` prefix:**

| Table | Model class | Purpose |
|-------|-------------|---------|
| `tab_generation_jobs` | `GenerationJob` | Synthesis job lifecycle |
| `tab_generation_profiles` | `GenerationProfile` | Reusable config profiles |
| `tab_multi_table_schemas` | `MultiTableSchema` | Multi-table relationship graphs |

## Module Descriptions

### `core/models.py`
SQLAlchemy ORM using `AumOSModel` (provides id, tenant_id, created_at, updated_at).
`GenerationJob.status` follows: `queued → training → generating → validating → complete/failed`.

### `core/interfaces.py`
Protocol classes only — no implementations. Import from here in `core/services.py`.
Never import from `adapters/` in `core/`.

### `core/services.py`
- `SchemaAnalysisService` — delegates to `SchemaAnalyzerProtocol`, stores analysis in DB
- `GenerationService` — full job lifecycle orchestration (analyze→train→generate→validate→export)
- `ProfileService` — CRUD for `GenerationProfile` entities
- `MultiTableService` — multi-table HMA synthesis preserving FK cardinalities
- `QualityGateService` — SDMetrics + Anonymeter evaluation, enforces threshold gates

### `adapters/generators/`
- `sdv_generator.py` — GaussianCopulaSynthesizer and TVAESynthesizer wrappers
- `ctgan_generator.py` — CTGANSynthesizer with configurable epochs, batch size, GPU
- `smartnoise_generator.py` — SmartNoise DP synthesizer; must call privacy engine for budget

### `adapters/schema_analyzer.py`
Uses pandas/polars to profile real data: detect categorical vs numeric, compute value
distributions, identify nullability, infer constraints, detect FK candidate columns.

### `adapters/constraint_engine.py`
Post-generation enforcement: clamp ranges, apply regex transforms, enforce unique sets,
resolve referential integrity by sampling from parent keys.

### `adapters/quality_gate.py`
Runs `sdmetrics.SingleTableMetadata`, `sdmetrics.QualityReport`, and `anonymeter`
`SinglingOutEvaluator` / `LinkabilityEvaluator`. Returns `fidelity_score` and `privacy_score`.

### `adapters/privacy_client.py`
HTTP client for `aumos-privacy-engine`. Calls `POST /api/v1/privacy/budget/allocate`
before each DP generation, `POST /api/v1/privacy/budget/consume` after.
Always validate that allocated epsilon ≤ tenant remaining budget before training.

### `adapters/storage.py`
MinIO client: bucket per tenant, object key `tab-jobs/{job_id}/output.{format}`.
Generates presigned GET URLs valid for 1 hour for download endpoints.

### `adapters/kafka.py`
Publishes to topic `aumos.tabular.generation.*`:
- `generation.job.queued`
- `generation.job.training_started`
- `generation.job.generating`
- `generation.job.validating`
- `generation.job.complete`
- `generation.job.failed`

### `api/schemas.py`
All request/response types as Pydantic v2 models with `model_config = ConfigDict(from_attributes=True)`.
Never return raw dicts from route handlers.

### `api/dependencies.py`
Use `request.app.state.*` to inject services into route handlers via `Depends()`.

## Coding Conventions

1. **Type hints** on every function signature, including return types
2. **Google-style docstrings** on all public classes and methods
3. **No `print()`** — always use `from aumos_common.observability import get_logger`
4. **No raw dicts** from route handlers — always Pydantic response models
5. **Async I/O** for all database, HTTP, Kafka, MinIO operations
6. **Hexagonal boundaries** — `core/` never imports from `adapters/`
7. **Structured log fields** — `logger.info("msg", job_id=str(job_id), tenant_id=str(...))`
8. **Error types** from `aumos_common.errors`: `AumOSError`, `NotFoundError`, `ConflictError`
9. **Pydantic validation** at all system boundaries (API input, external HTTP responses)
10. **RLS** — always pass `tenant_id` to repository methods; never query without it

## Job Lifecycle State Machine

```
QUEUED → TRAINING → GENERATING → VALIDATING → COMPLETE
                                             → FAILED (from any state)
```

Failed jobs must set `error_message`, `training_time_s` or `generation_time_s` (as applicable).

## DP Budget Protocol

For every SmartNoise generation job:
1. Call `privacy_client.allocate_budget(tenant_id, epsilon, delta)` → get `allocation_id`
2. If allocation fails (budget exceeded), transition job to FAILED immediately
3. After generation completes, call `privacy_client.consume_budget(allocation_id)`
4. Record the `allocation_id` in `GenerationJob.constraints` for audit trail

## Running Locally

```bash
pip install -e ".[dev]"
cp .env.example .env
docker compose -f docker-compose.dev.yml up -d postgres kafka redis minio
uvicorn aumos_tabular_engine.main:app --reload --port 8006
```

API docs: http://localhost:8006/docs
Health: http://localhost:8006/health

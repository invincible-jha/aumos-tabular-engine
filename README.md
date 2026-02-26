# aumos-tabular-engine

[![CI](https://github.com/aumos-enterprise/aumos-tabular-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/aumos-enterprise/aumos-tabular-engine/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![AumOS Enterprise](https://img.shields.io/badge/AumOS-Enterprise-purple.svg)](https://aumos.io)

Enterprise tabular synthetic data generation engine with differential privacy guarantees, multi-table relationship preservation, and comprehensive quality validation.

## Overview

`aumos-tabular-engine` generates high-fidelity synthetic tabular datasets for enterprise AI/ML workflows. It combines state-of-the-art generative models (CTGAN, TVAE, GaussianCopula) with rigorous differential privacy enforcement via SmartNoise-Synth, and validates output quality using SDMetrics and Anonymeter.

### Key Features

- **Multiple generators**: CTGAN (deep learning), TVAE, GaussianCopula (statistical)
- **Differential privacy**: SmartNoise DP with epsilon/delta budget management via `aumos-privacy-engine`
- **Multi-table synthesis**: Preserves foreign key relationships and referential integrity
- **Schema analysis**: Auto-detects column types, distributions, constraints, and relationships
- **Constraint enforcement**: Range, regex, not-null, unique, and referential integrity rules
- **Quality gate**: SDMetrics fidelity scores + Anonymeter privacy risk assessment
- **Enterprise multi-tenancy**: Row-level security with full tenant isolation
- **Kafka lifecycle events**: Every generation job emits structured events
- **Artifact storage**: MinIO-backed storage for generated datasets

## Architecture

```
aumos-tabular-engine
├── api/                    # FastAPI routers, Pydantic schemas, DI
│   ├── router.py           # REST endpoints
│   ├── schemas.py          # Request/response models
│   └── dependencies.py     # FastAPI dependency injection
├── core/                   # Domain logic (no framework imports)
│   ├── models.py           # SQLAlchemy ORM (tab_ prefix)
│   ├── interfaces.py       # Protocol classes
│   └── services.py         # Business logic services
└── adapters/               # Concrete implementations
    ├── generators/
    │   ├── sdv_generator.py       # GaussianCopula + TVAE
    │   ├── ctgan_generator.py     # CTGAN deep learning
    │   └── smartnoise_generator.py # DP-guaranteed generation
    ├── schema_analyzer.py   # Column type + distribution detection
    ├── constraint_engine.py # Business rule enforcement
    ├── quality_gate.py      # SDMetrics + Anonymeter
    ├── storage.py           # MinIO artifact client
    ├── kafka.py             # Lifecycle event publisher
    ├── privacy_client.py    # aumos-privacy-engine HTTP client
    └── repositories.py      # SQLAlchemy repositories
```

### Dependency Graph

```
aumos-common  ◄──── aumos-tabular-engine ────► aumos-privacy-engine
aumos-proto   ◄──┘                        └──► aumos-model-registry

aumos-fidelity-validator ◄── aumos-tabular-engine
aumos-data-pipeline      ◄── aumos-tabular-engine
aumos-marketplace        ◄── aumos-tabular-engine
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker + Docker Compose

### Local Development

```bash
# Clone and install
git clone https://github.com/aumos-enterprise/aumos-tabular-engine.git
cd aumos-tabular-engine
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env

# Start dependencies
docker compose -f docker-compose.dev.yml up -d postgres kafka redis minio

# Run the service
uvicorn aumos_tabular_engine.main:app --reload
```

### Using Docker Compose

```bash
docker compose -f docker-compose.dev.yml up -d
```

Service available at: `http://localhost:8006`

## API Reference

### Schema Analysis

```bash
# Analyze a dataset schema
POST /api/v1/tabular/analyze
Content-Type: application/json

{
  "source_uri": "s3://my-bucket/dataset.parquet",
  "sample_size": 10000
}

# Get analysis results
GET /api/v1/tabular/analyze/{job_id}
```

### Generation

```bash
# Start a generation job
POST /api/v1/tabular/generate
Content-Type: application/json

{
  "profile_id": "uuid",
  "num_rows": 100000,
  "model": "ctgan",
  "dp_epsilon": 1.0,
  "dp_delta": 1e-5,
  "constraints": [
    {"column": "age", "type": "range", "params": {"min": 18, "max": 99}},
    {"column": "email", "type": "regex", "params": {"pattern": "^[^@]+@[^@]+$"}}
  ],
  "output_format": "parquet"
}

# Check job status
GET /api/v1/tabular/generate/{job_id}

# Download generated data
GET /api/v1/tabular/generate/{job_id}/download
```

### Profiles

```bash
# Create a reusable generation profile
POST /api/v1/tabular/profiles
Content-Type: application/json

{
  "name": "customer-data-v1",
  "model_type": "ctgan",
  "default_dp_epsilon": 2.0,
  "default_constraints": [],
  "column_mappings": {}
}

# List all profiles
GET /api/v1/tabular/profiles

# Update a profile
PUT /api/v1/tabular/profiles/{id}
```

### Multi-Table

```bash
# Analyze multi-table schema
POST /api/v1/tabular/multi-table/analyze
Content-Type: application/json

{
  "tables": ["customers", "orders", "order_items"],
  "source_uri": "s3://my-bucket/schema.json"
}

# Generate linked tables
POST /api/v1/tabular/multi-table/generate
Content-Type: application/json

{
  "schema_id": "uuid",
  "num_rows_per_table": {"customers": 1000, "orders": 5000, "order_items": 20000},
  "model": "hma"
}
```

## Configuration

All configuration uses the `AUMOS_` prefix. See `.env.example` for all options.

| Variable | Default | Description |
|----------|---------|-------------|
| `AUMOS_SERVICE_NAME` | `aumos-tabular-engine` | Service identifier |
| `AUMOS_LOG_LEVEL` | `info` | Logging level |
| `AUMOS_DATABASE__URL` | — | PostgreSQL async URL |
| `AUMOS_KAFKA__BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka brokers |
| `AUMOS_REDIS__URL` | `redis://localhost:6379/0` | Redis URL |
| `AUMOS_MINIO__ENDPOINT` | `localhost:9000` | MinIO endpoint |
| `AUMOS_MINIO__ACCESS_KEY` | `minioadmin` | MinIO access key |
| `AUMOS_MINIO__SECRET_KEY` | `minioadmin` | MinIO secret key |
| `AUMOS_PRIVACY_ENGINE_URL` | `http://localhost:8010` | Privacy engine URL |
| `AUMOS_MODEL_REGISTRY_URL` | `http://localhost:8004` | Model registry URL |
| `AUMOS_GPU_ENABLED` | `false` | Enable GPU acceleration |

## Development Guide

```bash
make install      # install with dev deps
make test         # run full test suite
make test-quick   # fast subset (fail-fast)
make lint         # ruff lint + format check
make format       # auto-format with ruff
make typecheck    # mypy strict
make clean        # remove caches + build artifacts
make docker-build # build Docker image
make docker-run   # start docker-compose stack
```

## Related Repositories

| Repo | Role |
|------|------|
| [aumos-common](../aumos-common) | Shared utilities, base models, error handling |
| [aumos-proto](../aumos-proto) | Protobuf event schemas |
| [aumos-privacy-engine](../aumos-privacy-engine) | DP budget management |
| [aumos-model-registry](../aumos-model-registry) | Generator model tracking |
| [aumos-fidelity-validator](../aumos-fidelity-validator) | Downstream quality validation |
| [aumos-data-pipeline](../aumos-data-pipeline) | Orchestration and ingestion |
| [aumos-healthcare-synth](../aumos-healthcare-synth) | Healthcare-specific synthesis |

## License

Apache-2.0 — see [LICENSE](LICENSE).

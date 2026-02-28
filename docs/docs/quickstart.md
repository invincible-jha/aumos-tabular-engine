# Quickstart — 5-Minute Setup

Generate your first synthetic dataset from a real CSV in under 5 minutes.

## Prerequisites

- Docker and Docker Compose installed
- AumOS API key (contact your platform admin)
- Source dataset in Parquet or CSV format

## Demo Mode (No Dependencies)

The fastest path — no PostgreSQL, Kafka, or MinIO required:

```bash
git clone https://github.com/muveraai/aumos-tabular-engine
cd aumos-tabular-engine
docker compose -f docker-compose.demo.yml up
```

Wait ~30 seconds for startup, then generate synthetic data from the included demo dataset:

```bash
# Submit a generation job
curl -X POST http://localhost:8006/api/v1/generation/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "source_uri": "file://demo-data/customers.csv",
    "model": "ctgan",
    "num_rows": 1000,
    "output_format": "parquet"
  }'

# Response:
# {"job_id": "abc-123", "status": "queued", ...}
```

Poll for completion:

```bash
curl http://localhost:8006/api/v1/generation/jobs/abc-123
# {"status": "complete", "fidelity_score": 0.87, "output_uri": "..."}
```

Download the result:

```bash
curl http://localhost:8006/api/v1/generation/jobs/abc-123/download
# Returns presigned URL or file:// URI in demo mode
```

## Production Setup

For production, start all dependencies:

```bash
cp .env.example .env
# Edit .env with your PostgreSQL, Kafka, MinIO, and privacy engine URLs

docker compose -f docker-compose.dev.yml up -d postgres kafka redis minio
pip install -e ".[dev]"
uvicorn aumos_tabular_engine.main:app --reload --port 8006
```

See [Concepts: Generators](concepts/generators.md) for choosing the right generator for your use case.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added

- Initial scaffolding for aumos-tabular-engine
- FastAPI service with hexagonal architecture (api/core/adapters)
- SQLAlchemy models for GenerationJob, GenerationProfile, MultiTableSchema with `tab_` prefix
- SDV-based generators: GaussianCopula and TVAE via `sdv_generator` adapter
- CTGAN deep learning generator with configurable epochs and batch size
- SmartNoise differential privacy generator with epsilon/delta budget enforcement
- Schema analysis adapter: auto-detect column types, distributions, constraints, relationships
- Constraint engine: range, regex, not-null, unique, referential integrity enforcement
- Quality gate adapter: SDMetrics + Anonymeter evaluation pipeline
- Multi-table synthesis with foreign key preservation via `MultiTableService`
- MinIO adapter for dataset artifact storage with presigned URLs
- Kafka event publisher for generation lifecycle events
- HTTP client for aumos-privacy-engine budget allocation and tracking
- Full REST API: analyze, generate, profiles, multi-table endpoints
- Docker multi-stage build with non-root user and health check
- docker-compose.dev.yml with postgres, redis, kafka, minio, privacy-engine stub
- CI/CD pipeline: lint, typecheck, test, docker, license-check
- Standard deliverables: CLAUDE.md, README, pyproject.toml, Makefile, .env.example

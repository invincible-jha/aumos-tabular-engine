"""Business logic services for the AumOS Tabular Engine.

All services depend on repository and adapter interfaces (not concrete
implementations) and receive dependencies via constructor injection.
No framework code (FastAPI, SQLAlchemy) belongs here.
"""

import asyncio
import time
import uuid
from typing import Any

import pandas as pd

from aumos_common.errors import AumOSError, ConflictError, ErrorCode, NotFoundError
from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

from aumos_tabular_engine.core.interfaces import (
    ConstraintEngineProtocol,
    ConstraintSolverProtocol,
    EvaluationFrameworkProtocol,
    ExportHandlerProtocol,
    GeneratorProtocol,
    IGenerationJobRepository,
    IGenerationProfileRepository,
    IMultiTableSchemaRepository,
    MissingDataImputerProtocol,
    PrivacyWrapperProtocol,
    QualityGateProtocol,
    SchemaAnalyzerProtocol,
    StorageProtocol,
    SynthesisQualityEvaluatorProtocol,
)
from aumos_tabular_engine.core.models import GenerationJob, GenerationProfile, MultiTableSchema

logger = get_logger(__name__)

# Valid job status transitions
_VALID_STATUS_TRANSITIONS: dict[str, list[str]] = {
    "queued": ["training", "failed"],
    "training": ["generating", "failed"],
    "generating": ["validating", "failed"],
    "validating": ["complete", "failed"],
    "complete": [],
    "failed": [],
}

# Supported output format MIME types
_OUTPUT_MIME_TYPES: dict[str, str] = {
    "parquet": "application/octet-stream",
    "csv": "text/csv",
    "json": "application/json",
}


class SchemaAnalysisService:
    """Analyze real dataset schemas for synthetic data generation.

    Orchestrates schema detection by delegating to the analyzer adapter,
    then persists analysis results for use in generation jobs.
    """

    def __init__(
        self,
        analyzer: SchemaAnalyzerProtocol,
        job_repo: IGenerationJobRepository,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            analyzer: Schema analyzer adapter (pandas/polars-backed).
            job_repo: Job persistence repository.
            event_publisher: Kafka event publisher.
        """
        self._analyzer = analyzer
        self._job_repo = job_repo
        self._publisher = event_publisher

    async def analyze_dataframe(
        self,
        tenant_id: uuid.UUID,
        data: pd.DataFrame,
        sample_size: int | None = None,
    ) -> dict[str, Any]:
        """Analyze a DataFrame and return schema metadata.

        Runs column type detection, distribution profiling, and constraint
        inference on up to `sample_size` rows of the real data.

        Args:
            tenant_id: Requesting tenant for audit.
            data: Real source DataFrame to analyze.
            sample_size: Optional maximum rows to profile (defaults to all).

        Returns:
            Schema metadata dict with column types, distributions, and constraints.
        """
        logger.info(
            "Starting schema analysis",
            tenant_id=str(tenant_id),
            num_rows=len(data),
            num_cols=len(data.columns),
        )

        schema = await self._analyzer.analyze(data, sample_size)

        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.schema.analyzed",
                "tenant_id": str(tenant_id),
                "num_columns": len(data.columns),
                "num_rows_sampled": sample_size or len(data),
            },
        )

        logger.info(
            "Schema analysis complete",
            tenant_id=str(tenant_id),
            columns_detected=len(schema.get("columns", {})),
        )
        return schema


class GenerationService:
    """Orchestrate end-to-end synthetic data generation jobs.

    Manages the full lifecycle: schema analysis → model training →
    data generation → constraint application → quality validation → export.
    """

    def __init__(
        self,
        job_repo: IGenerationJobRepository,
        profile_repo: IGenerationProfileRepository,
        analyzer: SchemaAnalyzerProtocol,
        constraint_engine: ConstraintEngineProtocol,
        quality_gate: QualityGateProtocol,
        storage: StorageProtocol,
        event_publisher: EventPublisher,
        min_fidelity_score: float = 0.75,
        min_privacy_score: float = 0.50,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            job_repo: Generation job persistence.
            profile_repo: Profile persistence.
            analyzer: Schema analyzer adapter.
            constraint_engine: Business rule enforcer.
            quality_gate: Fidelity and privacy evaluator.
            storage: Artifact storage adapter.
            event_publisher: Kafka event publisher.
            min_fidelity_score: Minimum fidelity threshold to pass quality gate.
            min_privacy_score: Minimum privacy threshold to pass quality gate.
        """
        self._jobs = job_repo
        self._profiles = profile_repo
        self._analyzer = analyzer
        self._constraint_engine = constraint_engine
        self._quality_gate = quality_gate
        self._storage = storage
        self._publisher = event_publisher
        self._min_fidelity = min_fidelity_score
        self._min_privacy = min_privacy_score

    async def create_job(
        self,
        tenant_id: uuid.UUID,
        num_rows: int,
        model_type: str,
        profile_id: uuid.UUID | None = None,
        dp_epsilon: float | None = None,
        dp_delta: float | None = None,
        source_schema: dict[str, Any] | None = None,
        constraints: list[dict[str, Any]] | None = None,
        output_format: str = "parquet",
    ) -> GenerationJob:
        """Create a new generation job in QUEUED status.

        Validates model_type and output_format, merges profile defaults
        if a profile_id is provided, then persists the job.

        Args:
            tenant_id: Owning tenant.
            num_rows: Number of synthetic rows to generate.
            model_type: Generator backend to use.
            profile_id: Optional profile to inherit defaults from.
            dp_epsilon: Differential privacy epsilon (required for smartnoise).
            dp_delta: Differential privacy delta (required for smartnoise).
            source_schema: Pre-analyzed schema dict. If None, must be provided before running.
            constraints: Business rule constraints to enforce.
            output_format: Output file format (parquet/csv/json).

        Returns:
            Newly created GenerationJob in QUEUED status.

        Raises:
            NotFoundError: If profile_id given but profile not found.
            ConflictError: If model_type or output_format invalid.
        """
        valid_models = {"ctgan", "tvae", "gaussian_copula", "smartnoise", "hma"}
        if model_type not in valid_models:
            raise ConflictError(
                message=f"Invalid model_type '{model_type}'. Must be one of: {valid_models}",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        if output_format not in _OUTPUT_MIME_TYPES:
            raise ConflictError(
                message=f"Invalid output_format '{output_format}'. Must be: parquet, csv, json",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        if model_type == "smartnoise" and dp_epsilon is None:
            raise ConflictError(
                message="dp_epsilon is required when model_type='smartnoise'",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        # Merge profile defaults if provided
        merged_constraints = constraints or []
        if profile_id is not None:
            profile = await self._profiles.get_by_id(profile_id, tenant_id)
            if profile is None:
                raise NotFoundError(
                    message=f"Generation profile {profile_id} not found.",
                    error_code=ErrorCode.NOT_FOUND,
                )
            if not merged_constraints:
                merged_constraints = profile.default_constraints or []
            if dp_epsilon is None:
                dp_epsilon = profile.default_dp_epsilon
            if dp_delta is None:
                dp_delta = profile.default_dp_delta

        job = await self._jobs.create(
            tenant_id=tenant_id,
            model_type=model_type,
            num_rows=num_rows,
            profile_id=profile_id,
            dp_epsilon=dp_epsilon,
            dp_delta=dp_delta,
            source_schema=source_schema or {},
            constraints=merged_constraints,
            output_format=output_format,
        )

        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.job.queued",
                "tenant_id": str(tenant_id),
                "job_id": str(job.id),
                "model_type": model_type,
                "num_rows": num_rows,
            },
        )

        logger.info(
            "Generation job queued",
            tenant_id=str(tenant_id),
            job_id=str(job.id),
            model_type=model_type,
            num_rows=num_rows,
        )
        return job

    async def run_job(
        self,
        job_id: uuid.UUID,
        tenant_id: uuid.UUID,
        generator: GeneratorProtocol,
        real_data: pd.DataFrame,
    ) -> GenerationJob:
        """Execute a queued generation job end-to-end.

        Runs the full pipeline: analyze schema → train generator → generate data
        → apply constraints → evaluate quality → export artifact.

        Args:
            job_id: Job UUID to execute.
            tenant_id: Owning tenant.
            generator: Instantiated generator backend matching job.model_type.
            real_data: Source DataFrame to train the generator on.

        Returns:
            Updated GenerationJob with complete/failed status and all results.

        Raises:
            NotFoundError: If job not found.
            AumOSError: If generation or quality gate fails.
        """
        job = await self._jobs.get_by_id(job_id, tenant_id)
        if job is None:
            raise NotFoundError(
                message=f"Generation job {job_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        # Analyze schema if not already set
        source_schema = job.source_schema
        if not source_schema:
            source_schema = await self._analyzer.analyze(real_data)

        # --- TRAINING ---
        await self._jobs.update_status(job_id, "training")
        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.job.training_started",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
            },
        )

        training_start = time.monotonic()
        try:
            await generator.train(
                data=real_data,
                metadata=source_schema,
                epochs=None,  # use generator defaults
            )
        except Exception as exc:
            error_msg = f"Training failed: {exc}"
            await self._jobs.update_status(job_id, "failed", error_msg)
            await self._publisher.publish(
                Topics.DATA_SYNTHETIC,
                {
                    "event_type": "generation.job.failed",
                    "tenant_id": str(tenant_id),
                    "job_id": str(job_id),
                    "error": error_msg,
                },
            )
            logger.error("Generator training failed", job_id=str(job_id), error=error_msg)
            raise AumOSError(message=error_msg, error_code=ErrorCode.INTERNAL_ERROR) from exc

        training_time_s = time.monotonic() - training_start
        logger.info("Generator training complete", job_id=str(job_id), training_time_s=training_time_s)

        # --- GENERATION ---
        await self._jobs.update_status(job_id, "generating")
        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.job.generating",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
            },
        )

        generation_start = time.monotonic()
        try:
            synthetic_data = await generator.generate(num_rows=job.num_rows)
        except Exception as exc:
            error_msg = f"Generation failed: {exc}"
            await self._jobs.update_status(job_id, "failed", error_msg)
            await self._publisher.publish(
                Topics.DATA_SYNTHETIC,
                {
                    "event_type": "generation.job.failed",
                    "tenant_id": str(tenant_id),
                    "job_id": str(job_id),
                    "error": error_msg,
                },
            )
            raise AumOSError(message=error_msg, error_code=ErrorCode.INTERNAL_ERROR) from exc

        generation_time_s = time.monotonic() - generation_start

        # Apply business rule constraints
        if job.constraints:
            synthetic_data = self._constraint_engine.apply(synthetic_data, job.constraints)
            violations = self._constraint_engine.validate(synthetic_data, job.constraints)
            if violations:
                logger.warning(
                    "Constraint violations remain after apply",
                    job_id=str(job_id),
                    violations=violations[:5],
                )

        # --- VALIDATION ---
        await self._jobs.update_status(job_id, "validating")
        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.job.validating",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
            },
        )

        fidelity_score = await self._quality_gate.evaluate_fidelity(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=source_schema,
        )
        privacy_score = await self._quality_gate.evaluate_privacy(
            real_data=real_data,
            synthetic_data=synthetic_data,
        )

        logger.info(
            "Quality gate evaluation",
            job_id=str(job_id),
            fidelity_score=fidelity_score,
            privacy_score=privacy_score,
        )

        if fidelity_score < self._min_fidelity or privacy_score < self._min_privacy:
            error_msg = (
                f"Quality gate failed: fidelity={fidelity_score:.3f} "
                f"(min={self._min_fidelity}), privacy={privacy_score:.3f} "
                f"(min={self._min_privacy})"
            )
            await self._jobs.update_status(job_id, "failed", error_msg)
            await self._jobs.update_results(
                job_id=job_id,
                fidelity_score=fidelity_score,
                privacy_score=privacy_score,
                training_time_s=training_time_s,
                generation_time_s=generation_time_s,
            )
            await self._publisher.publish(
                Topics.DATA_SYNTHETIC,
                {
                    "event_type": "generation.job.failed",
                    "tenant_id": str(tenant_id),
                    "job_id": str(job_id),
                    "error": error_msg,
                },
            )
            raise AumOSError(message=error_msg, error_code=ErrorCode.QUALITY_GATE_FAILED)

        # Export artifact
        artifact_bytes = _serialize_dataframe(synthetic_data, job.output_format)
        filename = f"output.{job.output_format}"
        output_uri = await self._storage.upload(
            tenant_id=tenant_id,
            job_id=job_id,
            data=artifact_bytes,
            filename=filename,
            content_type=_OUTPUT_MIME_TYPES[job.output_format],
        )

        # --- COMPLETE ---
        job = await self._jobs.update_results(
            job_id=job_id,
            fidelity_score=fidelity_score,
            privacy_score=privacy_score,
            output_uri=output_uri,
            training_time_s=training_time_s,
            generation_time_s=generation_time_s,
        )
        await self._jobs.update_status(job_id, "complete")

        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.job.complete",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
                "fidelity_score": fidelity_score,
                "privacy_score": privacy_score,
                "output_uri": output_uri,
            },
        )

        logger.info(
            "Generation job complete",
            job_id=str(job_id),
            fidelity_score=fidelity_score,
            privacy_score=privacy_score,
            training_time_s=training_time_s,
            generation_time_s=generation_time_s,
        )
        return job

    async def get_job(
        self, job_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> GenerationJob:
        """Retrieve a generation job by ID.

        Args:
            job_id: Job UUID.
            tenant_id: Requesting tenant for isolation.

        Returns:
            GenerationJob ORM instance.

        Raises:
            NotFoundError: If job not found.
        """
        job = await self._jobs.get_by_id(job_id, tenant_id)
        if job is None:
            raise NotFoundError(
                message=f"Generation job {job_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return job

    async def list_jobs(
        self,
        tenant_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
    ) -> tuple[list[GenerationJob], int]:
        """List generation jobs for a tenant with pagination.

        Args:
            tenant_id: Requesting tenant.
            page: 1-based page number.
            page_size: Number of results per page.
            status: Optional status filter.

        Returns:
            Tuple of (jobs, total_count).
        """
        return await self._jobs.list_by_tenant(
            tenant_id=tenant_id,
            page=page,
            page_size=page_size,
            status=status,
        )

    async def get_download_url(
        self,
        job_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> str:
        """Generate a presigned download URL for a completed job's artifact.

        Args:
            job_id: Completed job UUID.
            tenant_id: Requesting tenant.

        Returns:
            Presigned URL valid for 1 hour.

        Raises:
            NotFoundError: If job not found or not complete.
            ConflictError: If job is not in 'complete' status.
        """
        job = await self.get_job(job_id, tenant_id)
        if job.status != "complete":
            raise ConflictError(
                message=f"Job {job_id} is not complete (status={job.status}). Cannot download.",
                error_code=ErrorCode.INVALID_OPERATION,
            )
        if not job.output_uri:
            raise ConflictError(
                message=f"Job {job_id} has no output artifact.",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        filename = f"output.{job.output_format}"
        return await self._storage.get_presigned_url(
            tenant_id=tenant_id,
            job_id=job_id,
            filename=filename,
        )


class ProfileService:
    """CRUD operations for generation profiles.

    Profiles store reusable generation configurations so users don't
    need to repeat parameters for recurring synthesis tasks.
    """

    def __init__(
        self,
        profile_repo: IGenerationProfileRepository,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            profile_repo: Profile persistence repository.
            event_publisher: Kafka event publisher.
        """
        self._profiles = profile_repo
        self._publisher = event_publisher

    async def create_profile(
        self,
        tenant_id: uuid.UUID,
        name: str,
        model_type: str,
        description: str | None = None,
        default_dp_epsilon: float | None = None,
        default_dp_delta: float | None = None,
        default_constraints: list[dict[str, Any]] | None = None,
        column_mappings: dict[str, Any] | None = None,
    ) -> GenerationProfile:
        """Create a new generation profile.

        Args:
            tenant_id: Owning tenant.
            name: Profile name unique within tenant.
            model_type: Default generator type.
            description: Optional description.
            default_dp_epsilon: Default epsilon for DP jobs.
            default_dp_delta: Default delta for DP jobs.
            default_constraints: Default constraint rules.
            column_mappings: Column type overrides.

        Returns:
            Newly created GenerationProfile.

        Raises:
            ConflictError: If a profile with the same name already exists.
        """
        existing = await self._profiles.list_by_tenant(tenant_id)
        if any(p.name == name for p in existing):
            raise ConflictError(
                message=f"Generation profile '{name}' already exists.",
                error_code=ErrorCode.ALREADY_EXISTS,
            )

        profile = await self._profiles.create(
            tenant_id=tenant_id,
            name=name,
            model_type=model_type,
            description=description,
            default_dp_epsilon=default_dp_epsilon,
            default_dp_delta=default_dp_delta,
            default_constraints=default_constraints or [],
            column_mappings=column_mappings or {},
        )

        logger.info(
            "Generation profile created",
            tenant_id=str(tenant_id),
            profile_id=str(profile.id),
            name=name,
        )
        return profile

    async def get_profile(
        self, profile_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> GenerationProfile:
        """Retrieve a profile by ID.

        Args:
            profile_id: Profile UUID.
            tenant_id: Requesting tenant.

        Returns:
            GenerationProfile ORM instance.

        Raises:
            NotFoundError: If profile not found.
        """
        profile = await self._profiles.get_by_id(profile_id, tenant_id)
        if profile is None:
            raise NotFoundError(
                message=f"Generation profile {profile_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return profile

    async def list_profiles(
        self, tenant_id: uuid.UUID, active_only: bool = True
    ) -> list[GenerationProfile]:
        """List all profiles for a tenant.

        Args:
            tenant_id: Requesting tenant.
            active_only: If True, exclude soft-deleted profiles.

        Returns:
            List of GenerationProfile instances.
        """
        return await self._profiles.list_by_tenant(tenant_id, active_only)

    async def update_profile(
        self,
        profile_id: uuid.UUID,
        tenant_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> GenerationProfile:
        """Apply partial updates to a profile.

        Args:
            profile_id: Profile UUID to update.
            tenant_id: Owning tenant.
            updates: Dict of field name → new value.

        Returns:
            Updated GenerationProfile.

        Raises:
            NotFoundError: If profile not found.
        """
        await self.get_profile(profile_id, tenant_id)  # validates existence
        profile = await self._profiles.update(profile_id, tenant_id, updates)
        logger.info(
            "Generation profile updated",
            profile_id=str(profile_id),
            tenant_id=str(tenant_id),
            updated_fields=list(updates.keys()),
        )
        return profile


class MultiTableService:
    """Multi-table synthesis with FK preservation.

    Orchestrates hierarchical multi-table generation using SDV's HMA
    (Hierarchical Modeling Algorithm) to preserve foreign key cardinalities
    and referential integrity across related tables.
    """

    def __init__(
        self,
        schema_repo: IMultiTableSchemaRepository,
        storage: StorageProtocol,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            schema_repo: Multi-table schema persistence.
            storage: Artifact storage adapter.
            event_publisher: Kafka event publisher.
        """
        self._schemas = schema_repo
        self._storage = storage
        self._publisher = event_publisher

    async def register_schema(
        self,
        tenant_id: uuid.UUID,
        name: str,
        tables: dict[str, Any],
        relationships: list[dict[str, Any]],
        source_uris: dict[str, str] | None = None,
        description: str | None = None,
    ) -> MultiTableSchema:
        """Register a multi-table schema for synthesis.

        Args:
            tenant_id: Owning tenant.
            name: Schema name unique within tenant.
            tables: Map of table_name → SDV SingleTableMetadata.
            relationships: FK relationships as list of {parent_table, parent_key, child_table, child_key}.
            source_uris: Map of table_name → MinIO URI for source data.
            description: Optional schema description.

        Returns:
            Newly registered MultiTableSchema.

        Raises:
            ConflictError: If a schema with the same name already exists.
        """
        existing = await self._schemas.list_by_tenant(tenant_id)
        if any(s.name == name for s in existing):
            raise ConflictError(
                message=f"Multi-table schema '{name}' already exists.",
                error_code=ErrorCode.ALREADY_EXISTS,
            )

        schema = await self._schemas.create(
            tenant_id=tenant_id,
            name=name,
            tables=tables,
            relationships=relationships,
            source_uris=source_uris or {},
            description=description,
        )

        logger.info(
            "Multi-table schema registered",
            tenant_id=str(tenant_id),
            schema_id=str(schema.id),
            table_count=len(tables),
            relationship_count=len(relationships),
        )
        return schema

    async def get_schema(
        self, schema_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> MultiTableSchema:
        """Retrieve a multi-table schema by ID.

        Args:
            schema_id: Schema UUID.
            tenant_id: Requesting tenant.

        Returns:
            MultiTableSchema ORM instance.

        Raises:
            NotFoundError: If schema not found.
        """
        schema = await self._schemas.get_by_id(schema_id, tenant_id)
        if schema is None:
            raise NotFoundError(
                message=f"Multi-table schema {schema_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return schema

    async def list_schemas(
        self, tenant_id: uuid.UUID
    ) -> list[MultiTableSchema]:
        """List all multi-table schemas for a tenant.

        Args:
            tenant_id: Requesting tenant.

        Returns:
            List of MultiTableSchema instances.
        """
        return await self._schemas.list_by_tenant(tenant_id)


class QualityGateService:
    """Run quality evaluation and enforce minimum thresholds.

    Wraps the quality gate adapter to provide structured evaluation
    results and enforce configured score thresholds.
    """

    def __init__(
        self,
        quality_gate: QualityGateProtocol,
        min_fidelity_score: float = 0.75,
        min_privacy_score: float = 0.50,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            quality_gate: SDMetrics + Anonymeter adapter.
            min_fidelity_score: Minimum fidelity to pass (0–1).
            min_privacy_score: Minimum privacy score to pass (0–1).
        """
        self._gate = quality_gate
        self._min_fidelity = min_fidelity_score
        self._min_privacy = min_privacy_score

    async def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
        key_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate both fidelity and privacy scores.

        Runs SDMetrics and Anonymeter evaluations in parallel and returns
        a structured result dict with scores, pass/fail status, and detail.

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Generated synthetic DataFrame.
            metadata: SDV-compatible column metadata.
            key_columns: Optional quasi-identifier columns for linkability tests.

        Returns:
            Dict with fidelity_score, privacy_score, passed, threshold violations.
        """
        fidelity_score, privacy_score = await asyncio.gather(
            self._gate.evaluate_fidelity(real_data, synthetic_data, metadata),
            self._gate.evaluate_privacy(real_data, synthetic_data, key_columns),
        )

        fidelity_passed = fidelity_score >= self._min_fidelity
        privacy_passed = privacy_score >= self._min_privacy
        overall_passed = fidelity_passed and privacy_passed

        result: dict[str, Any] = {
            "fidelity_score": fidelity_score,
            "privacy_score": privacy_score,
            "passed": overall_passed,
            "fidelity_passed": fidelity_passed,
            "privacy_passed": privacy_passed,
            "thresholds": {
                "min_fidelity": self._min_fidelity,
                "min_privacy": self._min_privacy,
            },
            "violations": [],
        }

        if not fidelity_passed:
            result["violations"].append(
                f"Fidelity score {fidelity_score:.3f} below threshold {self._min_fidelity}"
            )
        if not privacy_passed:
            result["violations"].append(
                f"Privacy score {privacy_score:.3f} below threshold {self._min_privacy}"
            )

        logger.info(
            "Quality gate evaluation complete",
            fidelity_score=fidelity_score,
            privacy_score=privacy_score,
            passed=overall_passed,
        )
        return result


def _serialize_dataframe(data: pd.DataFrame, output_format: str) -> bytes:
    """Serialize a DataFrame to the specified format.

    Args:
        data: DataFrame to serialize.
        output_format: Target format: parquet, csv, or json.

    Returns:
        Raw serialized bytes.

    Raises:
        ValueError: If output_format is not supported.
    """
    import io

    buffer = io.BytesIO()
    if output_format == "parquet":
        data.to_parquet(buffer, index=False, engine="pyarrow")
    elif output_format == "csv":
        buffer.write(data.to_csv(index=False).encode("utf-8"))
    elif output_format == "json":
        buffer.write(data.to_json(orient="records", lines=True).encode("utf-8"))
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# New domain-specific services wiring the 6 new adapters
# ---------------------------------------------------------------------------


class SynthesisQualityService:
    """Evaluate synthesis quality using SDMetrics and per-column diagnostics.

    Wraps the SynthesisQualityEvaluatorProtocol adapter to provide structured
    quality reports, threshold enforcement, and historical trend access.
    """

    def __init__(
        self,
        quality_evaluator: SynthesisQualityEvaluatorProtocol,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            quality_evaluator: SDMetrics-backed quality evaluator adapter.
            event_publisher: Kafka event publisher.
        """
        self._evaluator = quality_evaluator
        self._publisher = event_publisher

    async def evaluate_and_report(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
        include_diagnostics: bool = False,
    ) -> dict[str, Any]:
        """Run quality evaluation and publish results as a Kafka event.

        Args:
            tenant_id: Requesting tenant for audit.
            job_id: Generation job UUID.
            real_data: Original source DataFrame.
            synthetic_data: Synthetic DataFrame to evaluate.
            metadata: SDV-compatible metadata dict.
            include_diagnostics: If True, include per-column diagnostic detail.

        Returns:
            Quality evaluation result dict with scores and pass status.
        """
        logger.info(
            "Running synthesis quality evaluation",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
        )

        if include_diagnostics:
            result = await self._evaluator.get_diagnostic_report(
                real_data, synthetic_data, metadata
            )
        else:
            result = await self._evaluator.evaluate(real_data, synthetic_data, metadata)

        violations = self._evaluator.validate_thresholds(result)

        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.quality.evaluated",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
                "overall_score": result.get("overall_score"),
                "passed": result.get("passed"),
                "violation_count": len(violations),
            },
        )

        return {**result, "threshold_violations": violations}


class ConstraintSolverService:
    """Enforce relational constraints and business rules on synthetic DataFrames.

    Orchestrates validation, repair, and cross-table referential integrity
    enforcement using the ConstraintSolverProtocol adapter.
    """

    def __init__(
        self,
        constraint_solver: ConstraintSolverProtocol,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            constraint_solver: Constraint enforcement adapter.
            event_publisher: Kafka event publisher.
        """
        self._solver = constraint_solver
        self._publisher = event_publisher

    async def validate_and_repair(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        data: pd.DataFrame,
        constraints: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Validate constraints and iteratively repair violations.

        Args:
            tenant_id: Requesting tenant for audit.
            job_id: Generation job UUID.
            data: Synthetic DataFrame to validate and repair.
            constraints: List of constraint descriptor dicts.

        Returns:
            Tuple of (repaired_dataframe, violation_report).
        """
        initial_report = self._solver.report_violations(data, constraints)

        repaired_data = self._solver.apply(data, constraints)

        final_report = self._solver.report_violations(repaired_data, constraints)

        logger.info(
            "Constraint repair complete",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            initial_violations=initial_report["total_violations"],
            remaining_violations=final_report["total_violations"],
        )

        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.constraints.applied",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
                "initial_violations": initial_report["total_violations"],
                "remaining_violations": final_report["total_violations"],
            },
        )

        return repaired_data, final_report

    async def enforce_multi_table_integrity(
        self,
        tenant_id: uuid.UUID,
        tables: dict[str, pd.DataFrame],
        relationships: list[dict[str, str]],
    ) -> dict[str, pd.DataFrame]:
        """Enforce referential integrity across a set of synthesized tables.

        Args:
            tenant_id: Requesting tenant for audit.
            tables: Dict mapping table_name → synthetic DataFrame.
            relationships: FK relationship descriptors.

        Returns:
            Updated tables dict with all FK violations resolved.
        """
        result = self._solver.enforce_inter_table_referential_integrity(
            tables, relationships
        )
        logger.info(
            "Multi-table referential integrity enforced",
            tenant_id=str(tenant_id),
            table_count=len(tables),
            relationship_count=len(relationships),
        )
        return result


class MissingDataService:
    """Handle missing value imputation in real source data before synthesis.

    Orchestrates pattern analysis and imputation using the
    MissingDataImputerProtocol adapter. Intended for pre-processing real
    data prior to generator training, not for post-processing synthetic data.
    """

    def __init__(
        self,
        imputer: MissingDataImputerProtocol,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            imputer: Missing data imputation adapter.
            event_publisher: Kafka event publisher.
        """
        self._imputer = imputer
        self._publisher = event_publisher

    async def analyze_and_impute(
        self,
        tenant_id: uuid.UUID,
        data: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Analyze missing patterns and impute values in the given DataFrame.

        Args:
            tenant_id: Requesting tenant for audit.
            data: DataFrame to analyze and impute.
            metadata: Optional SDV-compatible metadata for column type hints.

        Returns:
            Tuple of (imputed_dataframe, analysis_report).
        """
        pattern_analysis = self._imputer.analyze_missing_patterns(data)

        logger.info(
            "Missing data analysis complete",
            tenant_id=str(tenant_id),
            affected_columns=len(pattern_analysis["affected_columns"]),
            total_missing=pattern_analysis["total_missing"],
        )

        if pattern_analysis["total_missing"] == 0:
            return data, {"imputation_performed": False, "analysis": pattern_analysis}

        imputed_data = await self._imputer.impute(data, metadata)
        quality_metrics = self._imputer.get_imputation_quality_metrics(data, imputed_data)

        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.data.imputed",
                "tenant_id": str(tenant_id),
                "columns_imputed": quality_metrics["columns_imputed"],
                "total_missing": pattern_analysis["total_missing"],
            },
        )

        report = {
            "imputation_performed": True,
            "analysis": pattern_analysis,
            "quality_metrics": quality_metrics,
        }
        return imputed_data, report


class PrivacyBudgetService:
    """Manage differential privacy budget allocation and monitoring.

    Coordinates epsilon budget requests, consumption, and tradeoff analysis
    using the PrivacyWrapperProtocol adapter.
    """

    def __init__(
        self,
        privacy_wrapper: PrivacyWrapperProtocol,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            privacy_wrapper: aumos-privacy-engine HTTP client adapter.
            event_publisher: Kafka event publisher.
        """
        self._wrapper = privacy_wrapper
        self._publisher = event_publisher

    async def request_budget_for_job(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        epsilon: float,
        delta: float,
    ) -> str:
        """Request DP budget allocation for a generation job.

        Validates the budget is sufficient before allocating. Transitions
        the job to FAILED if budget is exhausted.

        Args:
            tenant_id: Owning tenant.
            job_id: Generation job UUID.
            epsilon: Requested epsilon.
            delta: Requested delta.

        Returns:
            allocation_id string for use in budget consumption.

        Raises:
            ValueError: If the tenant has insufficient remaining budget.
        """
        validation = await self._wrapper.validate_epsilon_budget(tenant_id, epsilon)
        if not validation["sufficient"]:
            raise ValueError(
                f"Insufficient DP budget for tenant {tenant_id}: "
                f"requested={epsilon}, remaining={validation['remaining_budget']:.4f}"
            )

        allocation = await self._wrapper.allocate_budget(
            tenant_id=tenant_id,
            job_id=job_id,
            epsilon=epsilon,
            delta=delta,
        )

        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.privacy.budget_allocated",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
                "allocation_id": allocation.allocation_id,
                "epsilon": epsilon,
                "remaining_budget": allocation.remaining_budget,
            },
        )

        logger.info(
            "DP budget allocated",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            allocation_id=allocation.allocation_id,
            epsilon=epsilon,
        )
        return allocation.allocation_id

    async def finalize_budget_consumption(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        allocation_id: str,
        actual_epsilon_spent: float | None = None,
    ) -> None:
        """Finalize epsilon consumption after successful synthesis.

        Args:
            tenant_id: Owning tenant.
            job_id: Generation job UUID.
            allocation_id: Allocation ID from request_budget_for_job().
            actual_epsilon_spent: Actual epsilon consumed (if less than allocated).
        """
        await self._wrapper.consume_budget(allocation_id, actual_epsilon_spent)

        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.privacy.budget_consumed",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
                "allocation_id": allocation_id,
                "actual_epsilon_spent": actual_epsilon_spent,
            },
        )

        logger.info(
            "DP budget consumed",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            allocation_id=allocation_id,
        )

    async def get_budget_summary(
        self,
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Return a budget summary for the requesting tenant.

        Args:
            tenant_id: Requesting tenant.

        Returns:
            Dict with remaining, consumed, total budget and privacy level presets.
        """
        status = await self._wrapper.get_budget_status(tenant_id)
        return {
            "tenant_id": str(tenant_id),
            "remaining_budget": status.remaining_budget,
            "consumed_budget": status.consumed_budget,
            "total_budget": status.total_budget,
            "allocation_count": status.allocation_count,
        }


class ExportService:
    """Coordinate synthetic dataset export to multiple formats and storage.

    Wraps the ExportHandlerProtocol adapter to provide format selection,
    chunked export for large datasets, and presigned URL generation.
    """

    def __init__(
        self,
        export_handler: ExportHandlerProtocol,
        storage: StorageProtocol,
        event_publisher: EventPublisher,
        large_dataset_threshold_rows: int = 500_000,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            export_handler: Multi-format export adapter.
            storage: Artifact storage adapter.
            event_publisher: Kafka event publisher.
            large_dataset_threshold_rows: Row count above which chunked export is used.
        """
        self._handler = export_handler
        self._storage = storage
        self._publisher = event_publisher
        self._large_threshold = large_dataset_threshold_rows

    async def export_dataset(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        data: pd.DataFrame,
        output_format: str,
        **format_kwargs: Any,
    ) -> dict[str, Any]:
        """Export a synthetic dataset, using chunked export for large datasets.

        Automatically switches to chunked export when the DataFrame exceeds
        the configured large_dataset_threshold_rows.

        Args:
            tenant_id: Owning tenant.
            job_id: Generation job UUID.
            data: Synthetic DataFrame to export.
            output_format: Target format string.
            **format_kwargs: Format-specific serialization options.

        Returns:
            Export result dict with storage URI, bytes written, and metadata.
        """
        use_chunked = len(data) > self._large_threshold

        logger.info(
            "Exporting synthetic dataset",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            num_rows=len(data),
            output_format=output_format,
            chunked=use_chunked,
        )

        if use_chunked:
            result = await self._handler.export_chunked(
                data=data,
                output_format=output_format,
                tenant_id=tenant_id,
                job_id=job_id,
                **format_kwargs,
            )
        else:
            result = await self._handler.export(
                data=data,
                output_format=output_format,
                tenant_id=tenant_id,
                job_id=job_id,
                **format_kwargs,
            )

        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.export.complete",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
                "output_format": output_format,
                "num_rows": len(data),
                "chunked": use_chunked,
                "storage_uri": result.get("storage_uri"),
            },
        )

        return result

    async def get_download_url(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        filename: str,
        expires_seconds: int = 3600,
    ) -> str:
        """Generate a presigned download URL for an exported artifact.

        Args:
            tenant_id: Owning tenant.
            job_id: Generation job UUID.
            filename: Artifact filename to link.
            expires_seconds: URL validity duration.

        Returns:
            Presigned HTTPS URL for direct artifact download.
        """
        return await self._storage.get_presigned_url(
            tenant_id=tenant_id,
            job_id=job_id,
            filename=filename,
            expires_seconds=expires_seconds,
        )


class EvaluationService:
    """Orchestrate multi-objective evaluation and tradeoff reporting.

    Wraps the EvaluationFrameworkProtocol adapter to provide Pareto frontier
    computation, benchmark comparisons, and comprehensive evaluation reports.
    """

    def __init__(
        self,
        evaluation_framework: EvaluationFrameworkProtocol,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            evaluation_framework: Multi-objective evaluation adapter.
            event_publisher: Kafka event publisher.
        """
        self._framework = evaluation_framework
        self._publisher = event_publisher

    async def run_tradeoff_analysis(
        self,
        tenant_id: uuid.UUID,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        evaluation_points: list[Any],
        report_title: str = "Synthesis Evaluation Report",
    ) -> dict[str, Any]:
        """Run full tradeoff analysis and generate an evaluation report.

        Args:
            tenant_id: Requesting tenant for audit.
            real_data: Original source DataFrame.
            synthetic_data: Generated synthetic DataFrame.
            evaluation_points: EvaluationPoint instances collected from experiments.
            report_title: Human-readable report title.

        Returns:
            Comprehensive evaluation report dict.
        """
        report = await self._framework.generate_report(
            real_data=real_data,
            synthetic_data=synthetic_data,
            points=evaluation_points,
            report_title=report_title,
        )

        await self._publisher.publish(
            Topics.DATA_SYNTHETIC,
            {
                "event_type": "generation.evaluation.report_generated",
                "tenant_id": str(tenant_id),
                "total_points": report["evaluation_summary"]["total_points"],
                "pareto_points": report["evaluation_summary"]["pareto_points"],
                "has_recommendation": report["optimal_configuration"].get("recommendation") is not None,
            },
        )

        logger.info(
            "Evaluation report generated",
            tenant_id=str(tenant_id),
            total_points=report["evaluation_summary"]["total_points"],
        )
        return report

    async def get_optimal_configuration(
        self,
        evaluation_points: list[Any],
        require_production_ready: bool = True,
    ) -> dict[str, Any]:
        """Get the recommended optimal generator configuration.

        Args:
            evaluation_points: EvaluationPoint instances to consider.
            require_production_ready: Only consider points above quality thresholds.

        Returns:
            Recommendation dict with the best generator configuration.
        """
        return await self._framework.recommend_optimal_config(
            points=evaluation_points,
            require_production_ready=require_production_ready,
        )

    async def get_benchmark_comparison(
        self,
        evaluation_points: list[Any],
    ) -> dict[str, Any]:
        """Get a benchmark comparison across all evaluated generator types.

        Args:
            evaluation_points: EvaluationPoint instances to compare.

        Returns:
            Benchmark comparison dict with per-generator summaries and rankings.
        """
        return await self._framework.benchmark_generators(points=evaluation_points)

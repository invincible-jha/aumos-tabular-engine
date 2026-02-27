"""Protocol interfaces for the AumOS Tabular Engine.

All interfaces are defined as Python Protocol classes to enable
structural subtyping and clean dependency inversion. Adapters implement
these protocols; core services depend only on these abstractions.
"""

import uuid
from typing import Any, Protocol, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# New domain-specific adapter protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SynthesisQualityEvaluatorProtocol(Protocol):
    """Protocol for SDMetrics-backed quality evaluation.

    Evaluates column shape similarity, column pair trends, and aggregate
    quality scores between real and synthetic DataFrames.
    """

    async def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run full quality evaluation.

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Synthetic DataFrame to evaluate.
            metadata: SDV-compatible SingleTableMetadata dict.

        Returns:
            Evaluation report dict with overall_score, column_reports,
            degraded_columns, and passed flag.
        """
        ...

    async def get_diagnostic_report(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a per-column diagnostic report with recommendations.

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Synthetic DataFrame.
            metadata: SDV-compatible metadata dict.

        Returns:
            Diagnostic report with recommendations for degraded columns.
        """
        ...

    def validate_thresholds(self, evaluation_result: dict[str, Any]) -> list[str]:
        """Validate evaluation results against configured thresholds.

        Args:
            evaluation_result: Result dict from evaluate().

        Returns:
            List of threshold violation messages (empty if all pass).
        """
        ...


@runtime_checkable
class ConstraintSolverProtocol(Protocol):
    """Protocol for referential integrity and business rule constraint enforcement.

    Validates and iteratively repairs FK relationships, uniqueness constraints,
    value ranges, enum memberships, and cross-table referential integrity.
    """

    def validate(
        self,
        data: pd.DataFrame,
        constraints: list[dict[str, Any]],
    ) -> list[str]:
        """Validate synthetic data against a set of constraints.

        Args:
            data: Synthetic DataFrame to validate.
            constraints: List of constraint descriptor dicts.

        Returns:
            List of violation messages (empty if all constraints pass).
        """
        ...

    def apply(
        self,
        data: pd.DataFrame,
        constraints: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Apply constraints to synthetic data, repairing violations iteratively.

        Args:
            data: Synthetic DataFrame to transform.
            constraints: List of constraint descriptors.

        Returns:
            Transformed DataFrame with violations repaired.
        """
        ...

    def report_violations(
        self,
        data: pd.DataFrame,
        constraints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a structured violation report per constraint.

        Args:
            data: Synthetic DataFrame to inspect.
            constraints: List of constraint descriptors.

        Returns:
            Report dict with per-constraint pass/fail status.
        """
        ...

    def enforce_inter_table_referential_integrity(
        self,
        tables: dict[str, pd.DataFrame],
        relationships: list[dict[str, str]],
    ) -> dict[str, pd.DataFrame]:
        """Enforce FK relationships across multiple synthesized tables.

        Args:
            tables: Dict mapping table_name → DataFrame.
            relationships: FK relationship descriptors.

        Returns:
            Updated tables dict with all FK violations resolved.
        """
        ...


@runtime_checkable
class MissingDataImputerProtocol(Protocol):
    """Protocol for missing data imputation adapters.

    Handles KNN, MICE, and simple statistical imputation strategies
    with per-column strategy configuration support.
    """

    async def impute(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Impute all missing values in the DataFrame.

        Args:
            data: DataFrame with missing values to impute.
            metadata: Optional SDV-compatible metadata for column type hints.

        Returns:
            DataFrame with all missing values imputed.
        """
        ...

    def analyze_missing_patterns(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze missing value patterns in a DataFrame.

        Args:
            data: DataFrame to analyze.

        Returns:
            Missing pattern analysis dict with per-column breakdown.
        """
        ...

    def get_imputation_quality_metrics(
        self,
        original_data: pd.DataFrame,
        imputed_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Measure imputation quality comparing original and imputed DataFrames.

        Args:
            original_data: DataFrame before imputation (with NaNs).
            imputed_data: DataFrame after imputation (no NaNs).

        Returns:
            Quality metrics dict per column and aggregate statistics.
        """
        ...


@runtime_checkable
class PrivacyWrapperProtocol(Protocol):
    """Protocol for differential privacy budget management via aumos-privacy-engine.

    Handles epsilon budget allocation, consumption, monitoring, and
    synthesis-privacy tradeoff analysis via HTTP calls to the privacy engine.
    """

    async def allocate_budget(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        epsilon: float,
        delta: float,
        purpose: str = "tabular_synthesis",
    ) -> Any:
        """Request epsilon budget allocation from the privacy engine.

        Args:
            tenant_id: Owning tenant UUID.
            job_id: Generation job UUID.
            epsilon: Requested epsilon value.
            delta: Requested delta value.
            purpose: Human-readable purpose label for audit.

        Returns:
            BudgetAllocationResponse with allocation_id and status.

        Raises:
            ValueError: If the allocation is rejected due to budget exhaustion.
        """
        ...

    async def consume_budget(
        self,
        allocation_id: str,
        actual_epsilon_spent: float | None = None,
    ) -> dict[str, Any]:
        """Consume a previously allocated DP budget.

        Args:
            allocation_id: Allocation ID from a prior allocate_budget() call.
            actual_epsilon_spent: Optional actual epsilon consumed.

        Returns:
            Consumption confirmation dict.
        """
        ...

    async def get_budget_status(self, tenant_id: uuid.UUID) -> Any:
        """Query remaining DP budget for a tenant.

        Args:
            tenant_id: Tenant UUID to query.

        Returns:
            BudgetStatusResponse with remaining, consumed, and total budget.
        """
        ...

    async def validate_epsilon_budget(
        self,
        tenant_id: uuid.UUID,
        requested_epsilon: float,
    ) -> dict[str, Any]:
        """Check whether a tenant has sufficient budget for a requested epsilon.

        Args:
            tenant_id: Tenant UUID to check.
            requested_epsilon: Epsilon to validate against remaining budget.

        Returns:
            Dict with 'sufficient' bool and budget details.
        """
        ...


@runtime_checkable
class ExportHandlerProtocol(Protocol):
    """Protocol for synthetic dataset export to multiple file formats.

    Handles CSV, Parquet, Excel, and JSON exports with configurable
    options and uploads to MinIO/S3-compatible storage.
    """

    async def export(
        self,
        data: pd.DataFrame,
        output_format: str,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        filename: str | None = None,
        **format_kwargs: Any,
    ) -> dict[str, Any]:
        """Export a DataFrame in the specified format and upload to storage.

        Args:
            data: Synthetic DataFrame to export.
            output_format: Target format: csv, parquet, excel, json, jsonl.
            tenant_id: Owning tenant for storage path isolation.
            job_id: Generation job UUID for object key namespacing.
            filename: Override filename.
            **format_kwargs: Format-specific options.

        Returns:
            Export result dict with filename, storage_uri, and bytes_written.
        """
        ...

    async def export_chunked(
        self,
        data: pd.DataFrame,
        output_format: str,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        chunk_size: int | None = None,
        **format_kwargs: Any,
    ) -> dict[str, Any]:
        """Export large DataFrames in chunks to separate storage objects.

        Args:
            data: Large DataFrame to chunk and export.
            output_format: Target format for each chunk.
            tenant_id: Owning tenant UUID.
            job_id: Generation job UUID.
            chunk_size: Rows per chunk.
            **format_kwargs: Format-specific options.

        Returns:
            Chunked export manifest dict.
        """
        ...


@runtime_checkable
class EvaluationFrameworkProtocol(Protocol):
    """Protocol for multi-objective evaluation and tradeoff analysis.

    Computes Pareto frontiers, benchmark comparisons across generator types,
    optimal configuration recommendations, and comprehensive evaluation reports.
    """

    async def compute_pareto_frontier(
        self,
        points: list[Any] | None = None,
    ) -> list[Any]:
        """Compute Pareto-optimal evaluation points.

        Args:
            points: EvaluationPoints to analyze. Defaults to historical record.

        Returns:
            List of Pareto-optimal EvaluationPoints.
        """
        ...

    async def recommend_optimal_config(
        self,
        points: list[Any] | None = None,
        require_production_ready: bool = True,
    ) -> dict[str, Any]:
        """Recommend the optimal generator configuration.

        Args:
            points: Evaluation points to consider.
            require_production_ready: Only consider points above quality thresholds.

        Returns:
            Recommendation dict with the best configuration and scoring details.
        """
        ...

    async def benchmark_generators(
        self,
        points: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Compare generator types across evaluation dimensions.

        Args:
            points: Evaluation points. Defaults to historical record.

        Returns:
            Benchmark comparison dict with per-generator summaries and rankings.
        """
        ...

    async def generate_report(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        points: list[Any] | None = None,
        report_title: str = "Synthetic Data Evaluation Report",
    ) -> dict[str, Any]:
        """Generate a comprehensive evaluation report.

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Generated synthetic DataFrame.
            points: Evaluation points. Defaults to historical record.
            report_title: Human-readable report title.

        Returns:
            Full evaluation report dict.
        """
        ...


@runtime_checkable
class GeneratorProtocol(Protocol):
    """Protocol for synthetic data generators.

    All generator backends (CTGAN, TVAE, GaussianCopula, SmartNoise)
    must implement this interface.
    """

    async def train(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Train the generator on real data.

        Args:
            data: Real source DataFrame to learn from.
            metadata: SDV-compatible SingleTableMetadata dict describing columns.
            **kwargs: Generator-specific training hyperparameters.
        """
        ...

    async def generate(
        self,
        num_rows: int,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic rows.

        Args:
            num_rows: Number of synthetic rows to produce.
            conditions: Optional conditional generation constraints.

        Returns:
            DataFrame with `num_rows` synthetic rows matching the source schema.
        """
        ...

    def get_status(self) -> str:
        """Return the current generator status.

        Returns:
            One of: 'untrained', 'training', 'ready', 'error'.
        """
        ...


@runtime_checkable
class SchemaAnalyzerProtocol(Protocol):
    """Protocol for schema analysis adapters.

    Inspects real datasets to infer column types, distributions,
    nullability, constraints, and relationship candidates.
    """

    async def analyze(
        self,
        data: pd.DataFrame,
        sample_size: int | None = None,
    ) -> dict[str, Any]:
        """Analyze a DataFrame and return schema metadata.

        Args:
            data: DataFrame to profile.
            sample_size: Optional maximum rows to sample for profiling.

        Returns:
            Schema metadata dict compatible with SDV SingleTableMetadata,
            augmented with distribution summaries and detected constraints.
        """
        ...


@runtime_checkable
class ConstraintEngineProtocol(Protocol):
    """Protocol for post-generation constraint enforcement.

    Applies business rules to synthetic data to ensure domain validity
    beyond what the generative model can guarantee alone.
    """

    def validate(
        self,
        data: pd.DataFrame,
        constraints: list[dict[str, Any]],
    ) -> list[str]:
        """Validate synthetic data against a set of constraints.

        Args:
            data: Synthetic DataFrame to validate.
            constraints: List of constraint descriptors.

        Returns:
            List of validation error messages (empty if all pass).
        """
        ...

    def apply(
        self,
        data: pd.DataFrame,
        constraints: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Apply constraints to synthetic data, transforming where needed.

        Unlike validate(), this method modifies the DataFrame to satisfy
        constraints (e.g. clamping range violations, sampling from allowed sets).

        Args:
            data: Synthetic DataFrame to transform.
            constraints: List of constraint descriptors to enforce.

        Returns:
            Transformed DataFrame satisfying all constraints.
        """
        ...


@runtime_checkable
class QualityGateProtocol(Protocol):
    """Protocol for quality evaluation of synthetic datasets.

    Evaluates fidelity (how similar synthetic data is to real data)
    and privacy risk (how easily real individuals can be identified).
    """

    async def evaluate_fidelity(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> float:
        """Compute fidelity score using SDMetrics.

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Generated synthetic DataFrame.
            metadata: SDV-compatible metadata dict.

        Returns:
            Fidelity score in range [0.0, 1.0] where 1.0 = perfect fidelity.
        """
        ...

    async def evaluate_privacy(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        key_columns: list[str] | None = None,
    ) -> float:
        """Assess privacy risk using Anonymeter.

        Args:
            real_data: Original source DataFrame (held-out validation set).
            synthetic_data: Generated synthetic DataFrame.
            key_columns: Columns used as quasi-identifiers for linkability tests.

        Returns:
            Privacy score in range [0.0, 1.0] where 1.0 = maximum privacy protection.
        """
        ...


@runtime_checkable
class StorageProtocol(Protocol):
    """Protocol for artifact storage operations.

    Abstracts MinIO or S3-compatible object storage for generated datasets.
    """

    async def upload(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        data: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a generated dataset artifact.

        Args:
            tenant_id: Owning tenant for bucket isolation.
            job_id: Generation job UUID for object key namespacing.
            data: Raw artifact bytes to upload.
            filename: Target filename (e.g. 'output.parquet').
            content_type: MIME type of the artifact.

        Returns:
            Full MinIO URI for the uploaded object (s3://bucket/key).
        """
        ...

    async def download(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        filename: str,
    ) -> bytes:
        """Download a stored artifact.

        Args:
            tenant_id: Owning tenant.
            job_id: Generation job UUID.
            filename: Artifact filename to retrieve.

        Returns:
            Raw artifact bytes.
        """
        ...

    async def get_presigned_url(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        filename: str,
        expires_seconds: int = 3600,
    ) -> str:
        """Generate a presigned download URL for an artifact.

        Args:
            tenant_id: Owning tenant.
            job_id: Generation job UUID.
            filename: Artifact filename.
            expires_seconds: URL validity duration (default 1 hour).

        Returns:
            Presigned HTTPS URL valid for `expires_seconds`.
        """
        ...

    async def list_artifacts(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
    ) -> list[str]:
        """List all artifacts for a generation job.

        Args:
            tenant_id: Owning tenant.
            job_id: Generation job UUID.

        Returns:
            List of artifact filenames.
        """
        ...


class IGenerationJobRepository(Protocol):
    """Repository interface for GenerationJob persistence."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        model_type: str,
        num_rows: int,
        profile_id: uuid.UUID | None = None,
        dp_epsilon: float | None = None,
        dp_delta: float | None = None,
        source_schema: dict[str, Any] | None = None,
        constraints: list[dict[str, Any]] | None = None,
        output_format: str = "parquet",
    ) -> Any:
        """Create a new generation job in QUEUED status."""
        ...

    async def get_by_id(
        self, job_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Any | None:
        """Retrieve a job by ID with tenant scope."""
        ...

    async def update_status(
        self,
        job_id: uuid.UUID,
        status: str,
        error_message: str | None = None,
    ) -> Any:
        """Transition job to a new status."""
        ...

    async def update_results(
        self,
        job_id: uuid.UUID,
        fidelity_score: float | None = None,
        privacy_score: float | None = None,
        output_uri: str | None = None,
        training_time_s: float | None = None,
        generation_time_s: float | None = None,
    ) -> Any:
        """Store quality and timing results for a completed job."""
        ...

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
    ) -> tuple[list[Any], int]:
        """Paginated list of jobs for a tenant."""
        ...


class IGenerationProfileRepository(Protocol):
    """Repository interface for GenerationProfile persistence."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        name: str,
        model_type: str,
        description: str | None = None,
        default_dp_epsilon: float | None = None,
        default_dp_delta: float | None = None,
        default_constraints: list[dict[str, Any]] | None = None,
        column_mappings: dict[str, Any] | None = None,
    ) -> Any:
        """Create a new generation profile."""
        ...

    async def get_by_id(
        self, profile_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Any | None:
        """Retrieve a profile by ID with tenant scope."""
        ...

    async def list_by_tenant(
        self, tenant_id: uuid.UUID, active_only: bool = True
    ) -> list[Any]:
        """List all profiles for a tenant."""
        ...

    async def update(
        self,
        profile_id: uuid.UUID,
        tenant_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> Any:
        """Apply partial updates to a profile."""
        ...


class IMultiTableSchemaRepository(Protocol):
    """Repository interface for MultiTableSchema persistence."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        name: str,
        tables: dict[str, Any],
        relationships: list[dict[str, Any]],
        source_uris: dict[str, str] | None = None,
        description: str | None = None,
    ) -> Any:
        """Register a new multi-table schema."""
        ...

    async def get_by_id(
        self, schema_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Any | None:
        """Retrieve a schema by ID with tenant scope."""
        ...

    async def list_by_tenant(
        self, tenant_id: uuid.UUID
    ) -> list[Any]:
        """List all schemas for a tenant."""
        ...

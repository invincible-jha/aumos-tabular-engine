"""Protocol interfaces for the AumOS Tabular Engine.

All interfaces are defined as Python Protocol classes to enable
structural subtyping and clean dependency inversion. Adapters implement
these protocols; core services depend only on these abstractions.
"""

import uuid
from typing import Any, Protocol, runtime_checkable

import pandas as pd


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

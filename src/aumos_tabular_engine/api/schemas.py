"""Pydantic request and response schemas for the AumOS Tabular Engine API.

All schemas use strict validation and explicit field descriptions.
Never return raw dicts from route handlers — always use these models.
"""

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared / Constraint schemas
# ---------------------------------------------------------------------------


class ConstraintSchema(BaseModel):
    """A single business rule constraint to enforce on generated data.

    Attributes:
        column: Target column name to apply the constraint to.
        constraint_type: Type of constraint to apply.
        params: Type-specific constraint parameters.
    """

    model_config = ConfigDict(from_attributes=True)

    column: str = Field(..., description="Column name this constraint applies to")
    constraint_type: Literal["range", "regex", "not_null", "unique", "referential"] = Field(
        ..., description="Constraint type"
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific parameters (e.g. {min: 0, max: 100} for range)",
    )


# ---------------------------------------------------------------------------
# Schema Analysis schemas
# ---------------------------------------------------------------------------


class SchemaAnalysisRequest(BaseModel):
    """Request to analyze a dataset schema from a MinIO URI.

    Attributes:
        source_uri: MinIO URI to the source dataset (s3://bucket/key).
        sample_size: Optional row limit for profiling large datasets.
        detect_relationships: Whether to attempt FK relationship inference.
    """

    source_uri: str = Field(..., description="MinIO/S3 URI to the source dataset")
    sample_size: int | None = Field(
        default=None,
        ge=100,
        le=1_000_000,
        description="Maximum rows to sample for profiling (default: all rows)",
    )
    detect_relationships: bool = Field(
        default=False,
        description="Whether to infer foreign key relationships between columns",
    )


class ColumnProfileSchema(BaseModel):
    """Profile of a single column detected during schema analysis.

    Attributes:
        column_type: Detected SDV column type.
        sdtype: SDV semantic data type.
        null_fraction: Fraction of null values (0.0–1.0).
        unique_count: Estimated unique value count.
        distribution: Detected distribution summary.
    """

    model_config = ConfigDict(from_attributes=True)

    column_type: str = Field(..., description="SDV column type (numerical, categorical, datetime, etc.)")
    sdtype: str = Field(..., description="SDV semantic type")
    null_fraction: float = Field(..., ge=0.0, le=1.0, description="Fraction of null values")
    unique_count: int = Field(..., ge=0, description="Estimated unique value count")
    distribution: dict[str, Any] = Field(
        default_factory=dict,
        description="Distribution summary (mean, std, min, max, quantiles, top values)",
    )
    detected_constraints: list[ConstraintSchema] = Field(
        default_factory=list,
        description="Auto-inferred constraints for this column",
    )


class SchemaAnalysisResponse(BaseModel):
    """Result of schema analysis on a real dataset.

    Attributes:
        job_id: Analysis job ID for status tracking.
        num_rows: Number of rows in the source dataset.
        num_columns: Number of columns analyzed.
        columns: Per-column profile data.
        detected_relationships: Inferred FK relationships.
        metadata: Full SDV-compatible SingleTableMetadata dict.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: uuid.UUID = Field(..., description="Analysis job UUID")
    num_rows: int = Field(..., description="Total rows in source dataset")
    num_columns: int = Field(..., description="Number of columns analyzed")
    columns: dict[str, ColumnProfileSchema] = Field(
        default_factory=dict,
        description="Per-column profile keyed by column name",
    )
    detected_relationships: list[dict[str, str]] = Field(
        default_factory=list,
        description="Inferred FK relationships as {parent_column, child_column, cardinality}",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Full SDV-compatible SingleTableMetadata",
    )


# ---------------------------------------------------------------------------
# Generation Job schemas
# ---------------------------------------------------------------------------


class GenerationRequest(BaseModel):
    """Request to start a new synthetic data generation job.

    Attributes:
        profile_id: Optional profile to inherit defaults from.
        num_rows: Number of synthetic rows to generate.
        model: Generator model type.
        dp_epsilon: Differential privacy epsilon (required for smartnoise).
        dp_delta: Differential privacy delta.
        constraints: Business rule constraints to enforce.
        output_format: Output file format.
        source_schema: Pre-analyzed schema (or provide source_uri for auto-analysis).
        source_uri: MinIO URI to source data for on-the-fly analysis and training.
    """

    profile_id: uuid.UUID | None = Field(
        default=None,
        description="Optional generation profile to inherit defaults",
    )
    num_rows: int = Field(
        ...,
        ge=1,
        le=10_000_000,
        description="Number of synthetic rows to generate",
    )
    model: Literal["ctgan", "tvae", "gaussian_copula", "smartnoise", "hma"] = Field(
        default="ctgan",
        description="Generator backend",
    )
    dp_epsilon: float | None = Field(
        default=None,
        gt=0.0,
        le=100.0,
        description="Differential privacy epsilon (required when model=smartnoise)",
    )
    dp_delta: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Differential privacy delta",
    )
    constraints: list[ConstraintSchema] = Field(
        default_factory=list,
        description="Business rule constraints to enforce on generated data",
    )
    output_format: Literal["parquet", "csv", "json"] = Field(
        default="parquet",
        description="Output file format",
    )
    source_schema: dict[str, Any] | None = Field(
        default=None,
        description="Pre-analyzed SDV schema metadata (alternative to source_uri)",
    )
    source_uri: str | None = Field(
        default=None,
        description="MinIO URI for source data to train on",
    )


class GenerationJobResponse(BaseModel):
    """Response schema for a generation job.

    Attributes:
        job_id: Job UUID.
        tenant_id: Owning tenant.
        status: Current job status.
        model_type: Generator used.
        num_rows: Requested row count.
        fidelity_score: SDMetrics score (populated after completion).
        privacy_score: Anonymeter score (populated after completion).
        output_uri: MinIO artifact URI (populated after completion).
        training_time_s: Training duration (populated after completion).
        generation_time_s: Generation duration (populated after completion).
        error_message: Error detail if failed.
        created_at: Job creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: uuid.UUID = Field(..., description="Job UUID")
    tenant_id: uuid.UUID = Field(..., description="Owning tenant")
    status: str = Field(..., description="queued | training | generating | validating | complete | failed")
    model_type: str = Field(..., description="Generator backend used")
    num_rows: int = Field(..., description="Requested synthetic row count")
    output_format: str = Field(..., description="Output file format")
    fidelity_score: float | None = Field(None, description="SDMetrics fidelity score (0–1)")
    privacy_score: float | None = Field(None, description="Anonymeter privacy score (0–1)")
    output_uri: str | None = Field(None, description="MinIO artifact URI")
    training_time_s: float | None = Field(None, description="Training duration in seconds")
    generation_time_s: float | None = Field(None, description="Generation duration in seconds")
    error_message: str | None = Field(None, description="Error detail if status=failed")
    created_at: str | None = Field(None, description="ISO 8601 creation timestamp")


class JobListResponse(BaseModel):
    """Paginated list of generation jobs.

    Attributes:
        items: List of job summaries.
        total: Total job count matching filters.
        page: Current page number.
        page_size: Results per page.
    """

    items: list[GenerationJobResponse]
    total: int
    page: int
    page_size: int


class DownloadUrlResponse(BaseModel):
    """Presigned download URL for a completed job artifact.

    Attributes:
        job_id: Completed job UUID.
        download_url: Presigned URL valid for 1 hour.
        filename: Artifact filename.
        expires_in_seconds: URL validity duration.
    """

    job_id: uuid.UUID
    download_url: str
    filename: str
    expires_in_seconds: int = 3600


# ---------------------------------------------------------------------------
# Profile schemas
# ---------------------------------------------------------------------------


class ProfileCreate(BaseModel):
    """Request to create a new generation profile.

    Attributes:
        name: Profile name unique within tenant.
        model_type: Default generator type.
        description: Optional description.
        default_dp_epsilon: Default epsilon for DP jobs.
        default_dp_delta: Default delta for DP jobs.
        default_constraints: Default constraint rules.
        column_mappings: Column type overrides.
    """

    name: str = Field(..., min_length=1, max_length=255, description="Profile name")
    model_type: Literal["ctgan", "tvae", "gaussian_copula", "smartnoise", "hma"] = Field(
        default="ctgan",
        description="Default generator type for this profile",
    )
    description: str | None = Field(None, max_length=1000, description="Optional description")
    default_dp_epsilon: float | None = Field(None, gt=0.0, le=100.0)
    default_dp_delta: float | None = Field(None, gt=0.0, lt=1.0)
    default_constraints: list[ConstraintSchema] = Field(default_factory=list)
    column_mappings: dict[str, Any] = Field(default_factory=dict)


class ProfileUpdate(BaseModel):
    """Partial update request for a generation profile.

    All fields are optional; only provided fields are updated.
    """

    name: str | None = Field(None, min_length=1, max_length=255)
    model_type: Literal["ctgan", "tvae", "gaussian_copula", "smartnoise", "hma"] | None = None
    description: str | None = None
    default_dp_epsilon: float | None = Field(None, gt=0.0, le=100.0)
    default_dp_delta: float | None = Field(None, gt=0.0, lt=1.0)
    default_constraints: list[ConstraintSchema] | None = None
    column_mappings: dict[str, Any] | None = None
    is_active: bool | None = None


class ProfileResponse(BaseModel):
    """Response schema for a generation profile.

    Attributes:
        profile_id: Profile UUID.
        tenant_id: Owning tenant.
        name: Profile name.
        model_type: Default generator type.
        is_active: Whether the profile is active.
        created_at: Creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    profile_id: uuid.UUID = Field(..., alias="id", description="Profile UUID")
    tenant_id: uuid.UUID
    name: str
    model_type: str
    description: str | None = None
    default_dp_epsilon: float | None = None
    default_dp_delta: float | None = None
    default_constraints: list[ConstraintSchema] = Field(default_factory=list)
    column_mappings: dict[str, Any] = Field(default_factory=dict)
    is_active: bool
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Multi-table schemas
# ---------------------------------------------------------------------------


class MultiTableRelationship(BaseModel):
    """A foreign key relationship between two tables.

    Attributes:
        parent_table: Name of the parent (referenced) table.
        parent_key: Primary key column in the parent table.
        child_table: Name of the child (referencing) table.
        child_key: Foreign key column in the child table.
        cardinality: Relationship cardinality (1:N, N:N).
    """

    parent_table: str
    parent_key: str
    child_table: str
    child_key: str
    cardinality: Literal["1:N", "N:N"] = "1:N"


class MultiTableAnalysisRequest(BaseModel):
    """Request to analyze a multi-table schema.

    Attributes:
        name: Schema name to register.
        source_uris: Map of table_name → MinIO URI for each source table.
        relationships: FK relationships between tables (optional for auto-detect).
        description: Optional schema description.
    """

    name: str = Field(..., min_length=1, max_length=255)
    source_uris: dict[str, str] = Field(
        ...,
        min_length=2,
        description="Map of table_name → MinIO URI for source data",
    )
    relationships: list[MultiTableRelationship] = Field(
        default_factory=list,
        description="FK relationships (inferred if empty)",
    )
    description: str | None = None


class MultiTableGenerateRequest(BaseModel):
    """Request to generate a complete set of linked synthetic tables.

    Attributes:
        schema_id: Multi-table schema UUID to use.
        num_rows_per_table: Target row count per table.
        model: Multi-table generator (currently only 'hma' supported).
        output_format: Output format for all generated tables.
    """

    schema_id: uuid.UUID = Field(..., description="Multi-table schema UUID")
    num_rows_per_table: dict[str, int] = Field(
        ...,
        description="Map of table_name → desired synthetic row count",
    )
    model: Literal["hma"] = Field(default="hma", description="Multi-table generator (HMA)")
    output_format: Literal["parquet", "csv", "json"] = "parquet"


class MultiTableSchemaResponse(BaseModel):
    """Response schema for a multi-table schema.

    Attributes:
        schema_id: Schema UUID.
        tenant_id: Owning tenant.
        name: Schema name.
        table_count: Number of tables in the schema.
        relationship_count: Number of FK relationships.
        total_generation_jobs: Number of generation jobs run.
        created_at: Creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    schema_id: uuid.UUID = Field(..., alias="id")
    tenant_id: uuid.UUID
    name: str
    description: str | None = None
    table_count: int
    relationship_count: int
    total_generation_jobs: int = 0
    created_at: str | None = None

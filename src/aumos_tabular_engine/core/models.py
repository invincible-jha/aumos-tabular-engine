"""SQLAlchemy ORM models for the AumOS Tabular Engine.

All tables use the `tab_` prefix. Tenant-scoped tables extend AumOSModel
which supplies id (UUID), tenant_id, created_at, and updated_at columns.
"""

import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aumos_common.database import AumOSModel


class GenerationJob(AumOSModel):
    """A synthetic data generation job.

    Tracks the full lifecycle of a generation request from queuing through
    training, generation, validation, and completion or failure.

    Status transitions: queued → training → generating → validating → complete/failed

    Table: tab_generation_jobs
    """

    __tablename__ = "tab_generation_jobs"

    profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tab_generation_profiles.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Optional linked generation profile",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="queued",
        index=True,
        comment="queued | training | generating | validating | complete | failed",
    )
    model_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="ctgan | tvae | gaussian_copula | smartnoise | hma",
    )
    num_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of synthetic rows to generate",
    )
    dp_epsilon: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Differential privacy epsilon budget for this job",
    )
    dp_delta: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Differential privacy delta parameter",
    )
    source_schema: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Inferred source schema from schema analysis",
    )
    constraints: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="Business rule constraints to enforce post-generation",
    )
    fidelity_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="SDMetrics overall fidelity score (0–1)",
    )
    privacy_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Anonymeter privacy risk score (0–1, higher = more private)",
    )
    output_uri: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="MinIO URI for the generated dataset artifact",
    )
    output_format: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="parquet",
        comment="parquet | csv | json",
    )
    training_time_s: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Wall-clock training duration in seconds",
    )
    generation_time_s: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Wall-clock generation duration in seconds",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error detail if status=failed",
    )
    dp_allocation_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Privacy engine allocation ID for DP budget tracking",
    )

    profile: Mapped["GenerationProfile | None"] = relationship(
        "GenerationProfile",
        back_populates="jobs",
    )


class GenerationProfile(AumOSModel):
    """A reusable configuration profile for synthetic data generation.

    Profiles store default generation parameters and column mappings
    so users don't need to repeat configuration for recurring tasks.

    Table: tab_generation_profiles
    """

    __tablename__ = "tab_generation_profiles"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_tab_profiles_tenant_name"),
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Human-readable profile name unique within tenant",
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="ctgan",
        comment="Default generator model type for this profile",
    )
    default_dp_epsilon: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Default epsilon for DP-enabled generation",
    )
    default_dp_delta: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Default delta for DP-enabled generation",
    )
    default_constraints: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="Default constraint rules applied to all jobs using this profile",
    )
    column_mappings: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Column type overrides and semantic mappings",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Soft-delete flag — inactive profiles cannot be used for new jobs",
    )

    jobs: Mapped[list["GenerationJob"]] = relationship(
        "GenerationJob",
        back_populates="profile",
    )


class MultiTableSchema(AumOSModel):
    """A registered multi-table schema for hierarchical synthesis.

    Captures the table definitions, column types, and foreign key relationships
    needed to generate a coherent multi-table synthetic dataset that preserves
    referential integrity.

    Table: tab_multi_table_schemas
    """

    __tablename__ = "tab_multi_table_schemas"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "name", name="uq_tab_mt_schemas_tenant_name"
        ),
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Human-readable schema name unique within tenant",
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tables: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Map of table_name → SDV SingleTableMetadata dict",
    )
    relationships: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of FK relationships: {parent_table, parent_key, child_table, child_key}",
    )
    source_uris: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Map of table_name → MinIO URI for source data",
    )
    row_counts: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Map of table_name → detected row count in source",
    )
    analysis_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="GenerationJob ID of the schema analysis run",
    )
    total_generation_jobs: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Count of generation jobs run against this schema",
    )

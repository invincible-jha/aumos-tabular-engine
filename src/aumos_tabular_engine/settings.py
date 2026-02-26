"""Tabular Engine service settings extending AumOS base configuration."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class MinIOSettings(AumOSSettings):
    """MinIO connection configuration.

    Attributes:
        endpoint: MinIO server endpoint (host:port).
        access_key: MinIO access key credential.
        secret_key: MinIO secret key credential.
        secure: Whether to use TLS for MinIO connection.
        bucket_prefix: Prefix for tabular engine buckets.
    """

    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    secure: bool = False
    bucket_prefix: str = "tab-artifacts"

    model_config = SettingsConfigDict(env_prefix="AUMOS_MINIO__")


class Settings(AumOSSettings):
    """Configuration for the AumOS Tabular Engine service.

    Extends base AumOS settings with tabular-engine-specific configuration
    for generator backends, privacy engine integration, and artifact storage.
    """

    service_name: str = "aumos-tabular-engine"

    # External service URLs
    privacy_engine_url: str = Field(
        default="http://localhost:8010",
        description="Base URL for aumos-privacy-engine",
    )
    model_registry_url: str = Field(
        default="http://localhost:8004",
        description="Base URL for aumos-model-registry",
    )

    # GPU acceleration
    gpu_enabled: bool = Field(
        default=False,
        description="Enable GPU acceleration for CTGAN and TVAE generators",
    )

    # CTGAN defaults
    ctgan_default_epochs: int = Field(
        default=300,
        description="Default training epochs for CTGAN generator",
    )
    ctgan_default_batch_size: int = Field(
        default=500,
        description="Default batch size for CTGAN training",
    )

    # Quality gate thresholds
    min_fidelity_score: float = Field(
        default=0.75,
        description="Minimum acceptable fidelity score (0–1) to pass quality gate",
    )
    min_privacy_score: float = Field(
        default=0.50,
        description="Minimum acceptable privacy score (0–1) to pass quality gate",
    )

    # DP defaults
    default_dp_epsilon: float = Field(
        default=1.0,
        description="Default epsilon for differential privacy (lower = more private)",
    )
    default_dp_delta: float = Field(
        default=1e-5,
        description="Default delta for differential privacy",
    )

    # Artifact storage
    artifact_bucket: str = Field(
        default="tab-artifacts",
        description="MinIO bucket name for generated dataset artifacts",
    )

    # HTTP client timeouts (seconds)
    http_timeout: float = Field(
        default=30.0,
        description="Timeout for HTTP calls to downstream services",
    )

    model_config = SettingsConfigDict(env_prefix="AUMOS_")

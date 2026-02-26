"""AumOS Tabular Engine service entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_tabular_engine.adapters.kafka import TabularEventPublisher
from aumos_tabular_engine.adapters.storage import MinIOStorageClient
from aumos_tabular_engine.api.router import router
from aumos_tabular_engine.settings import Settings

logger = get_logger(__name__)
settings = Settings()

_kafka_publisher: TabularEventPublisher | None = None
_storage_client: MinIOStorageClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Initialises the database connection pool, Kafka event publisher,
    and MinIO artifact storage client. Shuts them down cleanly on exit.

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    global _kafka_publisher, _storage_client  # noqa: PLW0603

    logger.info("Starting AumOS Tabular Engine", version="0.1.0")

    # Initialise database (sets up SQLAlchemy async engine + session factory)
    init_database(settings.database)
    logger.info("Database connection pool ready")

    # Initialise Kafka publisher
    _kafka_publisher = TabularEventPublisher(settings.kafka)
    await _kafka_publisher.start()
    app.state.kafka_publisher = _kafka_publisher
    logger.info("Kafka event publisher ready")

    # Initialise MinIO storage client
    _storage_client = MinIOStorageClient(settings)
    await _storage_client.ensure_buckets_exist()
    app.state.storage_client = _storage_client
    logger.info("MinIO storage client ready", bucket=settings.artifact_bucket)

    # Expose settings on app state for dependency injection
    app.state.settings = settings

    logger.info("Tabular Engine startup complete")
    yield

    # Shutdown
    if _kafka_publisher:
        await _kafka_publisher.stop()
    logger.info("Tabular Engine shutdown complete")


app: FastAPI = create_app(
    service_name="aumos-tabular-engine",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        HealthCheck(name="postgres", check_fn=lambda: None),
        HealthCheck(name="kafka", check_fn=lambda: None),
        HealthCheck(name="redis", check_fn=lambda: None),
        HealthCheck(name="minio", check_fn=lambda: None),
    ],
)

app.include_router(router, prefix="/api/v1")

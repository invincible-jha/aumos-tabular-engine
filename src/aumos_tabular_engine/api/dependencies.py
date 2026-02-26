"""FastAPI dependency injection for the AumOS Tabular Engine.

All route handlers receive services via FastAPI Depends() rather than
constructing them directly. This file wires together adapters and services
from app.state.
"""

from fastapi import Depends, Request

from aumos_tabular_engine.adapters.constraint_engine import ConstraintEngine
from aumos_tabular_engine.adapters.kafka import TabularEventPublisher
from aumos_tabular_engine.adapters.privacy_client import PrivacyEngineClient
from aumos_tabular_engine.adapters.quality_gate import QualityGateAdapter
from aumos_tabular_engine.adapters.repositories import (
    GenerationJobRepository,
    GenerationProfileRepository,
    MultiTableSchemaRepository,
)
from aumos_tabular_engine.adapters.schema_analyzer import SchemaAnalyzer
from aumos_tabular_engine.adapters.storage import MinIOStorageClient
from aumos_tabular_engine.core.services import (
    GenerationService,
    MultiTableService,
    ProfileService,
    QualityGateService,
    SchemaAnalysisService,
)
from aumos_tabular_engine.settings import Settings


def get_settings(request: Request) -> Settings:
    """Provide configured Settings from app state.

    Args:
        request: Current FastAPI request.

    Returns:
        Service Settings instance.
    """
    return request.app.state.settings  # type: ignore[no-any-return]


def get_kafka_publisher(request: Request) -> TabularEventPublisher:
    """Provide the Kafka event publisher from app state.

    Args:
        request: Current FastAPI request.

    Returns:
        TabularEventPublisher singleton.
    """
    return request.app.state.kafka_publisher  # type: ignore[no-any-return]


def get_storage_client(request: Request) -> MinIOStorageClient:
    """Provide the MinIO storage client from app state.

    Args:
        request: Current FastAPI request.

    Returns:
        MinIOStorageClient singleton.
    """
    return request.app.state.storage_client  # type: ignore[no-any-return]


def get_job_repo() -> GenerationJobRepository:
    """Provide a GenerationJobRepository instance.

    Returns:
        GenerationJobRepository constructed with the shared DB session factory.
    """
    return GenerationJobRepository()


def get_profile_repo() -> GenerationProfileRepository:
    """Provide a GenerationProfileRepository instance.

    Returns:
        GenerationProfileRepository constructed with the shared DB session factory.
    """
    return GenerationProfileRepository()


def get_schema_repo() -> MultiTableSchemaRepository:
    """Provide a MultiTableSchemaRepository instance.

    Returns:
        MultiTableSchemaRepository constructed with the shared DB session factory.
    """
    return MultiTableSchemaRepository()


def get_generation_service(
    request: Request,
    job_repo: GenerationJobRepository = Depends(get_job_repo),
    profile_repo: GenerationProfileRepository = Depends(get_profile_repo),
    kafka_publisher: TabularEventPublisher = Depends(get_kafka_publisher),
    storage_client: MinIOStorageClient = Depends(get_storage_client),
    settings: Settings = Depends(get_settings),
) -> GenerationService:
    """Provide a fully-wired GenerationService.

    Args:
        request: Current FastAPI request.
        job_repo: Job persistence repository.
        profile_repo: Profile persistence repository.
        kafka_publisher: Kafka event publisher.
        storage_client: MinIO artifact storage.
        settings: Service configuration.

    Returns:
        GenerationService with all dependencies injected.
    """
    analyzer = SchemaAnalyzer()
    constraint_engine = ConstraintEngine()
    quality_gate = QualityGateAdapter()

    return GenerationService(
        job_repo=job_repo,
        profile_repo=profile_repo,
        analyzer=analyzer,
        constraint_engine=constraint_engine,
        quality_gate=quality_gate,
        storage=storage_client,
        event_publisher=kafka_publisher,
        min_fidelity_score=settings.min_fidelity_score,
        min_privacy_score=settings.min_privacy_score,
    )


def get_profile_service(
    kafka_publisher: TabularEventPublisher = Depends(get_kafka_publisher),
    profile_repo: GenerationProfileRepository = Depends(get_profile_repo),
) -> ProfileService:
    """Provide a fully-wired ProfileService.

    Args:
        kafka_publisher: Kafka event publisher.
        profile_repo: Profile persistence repository.

    Returns:
        ProfileService with all dependencies injected.
    """
    return ProfileService(
        profile_repo=profile_repo,
        event_publisher=kafka_publisher,
    )


def get_multi_table_service(
    kafka_publisher: TabularEventPublisher = Depends(get_kafka_publisher),
    storage_client: MinIOStorageClient = Depends(get_storage_client),
    schema_repo: MultiTableSchemaRepository = Depends(get_schema_repo),
) -> MultiTableService:
    """Provide a fully-wired MultiTableService.

    Args:
        kafka_publisher: Kafka event publisher.
        storage_client: MinIO artifact storage.
        schema_repo: Multi-table schema persistence.

    Returns:
        MultiTableService with all dependencies injected.
    """
    return MultiTableService(
        schema_repo=schema_repo,
        storage=storage_client,
        event_publisher=kafka_publisher,
    )


def get_schema_analysis_service(
    kafka_publisher: TabularEventPublisher = Depends(get_kafka_publisher),
    job_repo: GenerationJobRepository = Depends(get_job_repo),
) -> SchemaAnalysisService:
    """Provide a fully-wired SchemaAnalysisService.

    Args:
        kafka_publisher: Kafka event publisher.
        job_repo: Job persistence repository.

    Returns:
        SchemaAnalysisService with all dependencies injected.
    """
    analyzer = SchemaAnalyzer()
    return SchemaAnalysisService(
        analyzer=analyzer,
        job_repo=job_repo,
        event_publisher=kafka_publisher,
    )


def get_quality_gate_service(
    settings: Settings = Depends(get_settings),
) -> QualityGateService:
    """Provide a fully-wired QualityGateService.

    Args:
        settings: Service configuration for score thresholds.

    Returns:
        QualityGateService with configured thresholds.
    """
    quality_gate = QualityGateAdapter()
    return QualityGateService(
        quality_gate=quality_gate,
        min_fidelity_score=settings.min_fidelity_score,
        min_privacy_score=settings.min_privacy_score,
    )


def get_privacy_client(
    settings: Settings = Depends(get_settings),
) -> PrivacyEngineClient:
    """Provide a PrivacyEngineClient HTTP client.

    Args:
        settings: Service configuration with privacy engine URL.

    Returns:
        PrivacyEngineClient instance.
    """
    return PrivacyEngineClient(
        base_url=settings.privacy_engine_url,
        timeout=settings.http_timeout,
    )

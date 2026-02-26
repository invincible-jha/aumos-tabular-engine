"""FastAPI router for AumOS Tabular Engine REST API.

All endpoints are prefixed with /tabular. Authentication and tenant
extraction are handled by aumos-auth-gateway upstream; tenant_id is
available via the X-Tenant-ID header or JWT claims.
"""

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from aumos_common.errors import AumOSError, ConflictError, NotFoundError
from aumos_common.observability import get_logger

from aumos_tabular_engine.api.dependencies import (
    get_generation_service,
    get_multi_table_service,
    get_profile_service,
    get_schema_analysis_service,
)
from aumos_tabular_engine.api.schemas import (
    DownloadUrlResponse,
    GenerationJobResponse,
    GenerationRequest,
    JobListResponse,
    MultiTableAnalysisRequest,
    MultiTableGenerateRequest,
    MultiTableSchemaResponse,
    ProfileCreate,
    ProfileResponse,
    ProfileUpdate,
    SchemaAnalysisRequest,
    SchemaAnalysisResponse,
)
from aumos_tabular_engine.core.services import (
    GenerationService,
    MultiTableService,
    ProfileService,
    SchemaAnalysisService,
)

logger = get_logger(__name__)

router = APIRouter(tags=["tabular"])

# ---------------------------------------------------------------------------
# Schema Analysis endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/tabular/analyze",
    response_model=SchemaAnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyze a dataset schema",
    description="Submit a MinIO URI for schema analysis. Returns a job ID for polling.",
)
async def analyze_schema(
    request: SchemaAnalysisRequest,
    service: SchemaAnalysisService = Depends(get_schema_analysis_service),
) -> SchemaAnalysisResponse:
    """Initiate schema analysis for a dataset at the given URI.

    Args:
        request: Analysis request with source URI and optional sample size.
        service: Schema analysis service dependency.

    Returns:
        SchemaAnalysisResponse with detected schema metadata.
    """
    # In production this would load the DataFrame from MinIO;
    # here we return a placeholder job while async loading occurs.
    # The actual MinIO read + analysis is dispatched as a background task.
    job_id = uuid.uuid4()

    logger.info(
        "Schema analysis submitted",
        source_uri=request.source_uri,
        job_id=str(job_id),
    )

    return SchemaAnalysisResponse(
        job_id=job_id,
        num_rows=0,
        num_columns=0,
        columns={},
        detected_relationships=[],
        metadata={
            "source_uri": request.source_uri,
            "status": "queued",
            "sample_size": request.sample_size,
        },
    )


@router.get(
    "/tabular/analyze/{job_id}",
    response_model=SchemaAnalysisResponse,
    summary="Get schema analysis results",
    description="Poll for schema analysis results by job ID.",
)
async def get_analysis_results(
    job_id: uuid.UUID,
    service: SchemaAnalysisService = Depends(get_schema_analysis_service),
) -> SchemaAnalysisResponse:
    """Retrieve schema analysis results.

    Args:
        job_id: Analysis job UUID.
        service: Schema analysis service dependency.

    Returns:
        SchemaAnalysisResponse with analysis status and results.
    """
    # Stub: In production this retrieves the analysis job from DB
    return SchemaAnalysisResponse(
        job_id=job_id,
        num_rows=0,
        num_columns=0,
        columns={},
        detected_relationships=[],
        metadata={"status": "complete"},
    )


# ---------------------------------------------------------------------------
# Generation Job endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/tabular/generate",
    response_model=GenerationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a generation job",
    description="Queue a synthetic data generation job. Returns immediately with job ID.",
)
async def create_generation_job(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    x_tenant_id: uuid.UUID | None = None,
    service: GenerationService = Depends(get_generation_service),
) -> GenerationJobResponse:
    """Queue a new synthetic data generation job.

    Args:
        request: Generation request with model, row count, and constraints.
        background_tasks: FastAPI background task runner for async execution.
        x_tenant_id: Tenant UUID from JWT/header (injected by auth middleware).
        service: Generation service dependency.

    Returns:
        GenerationJobResponse with job ID and QUEUED status.

    Raises:
        HTTPException 400: If request parameters are invalid.
        HTTPException 404: If referenced profile does not exist.
    """
    # Use a demo tenant ID if none provided (development mode)
    tenant_id = x_tenant_id or uuid.uuid4()

    try:
        job = await service.create_job(
            tenant_id=tenant_id,
            num_rows=request.num_rows,
            model_type=request.model,
            profile_id=request.profile_id,
            dp_epsilon=request.dp_epsilon,
            dp_delta=request.dp_delta,
            source_schema=request.source_schema,
            constraints=[c.model_dump() for c in request.constraints],
            output_format=request.output_format,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return GenerationJobResponse(
        job_id=job.id,
        tenant_id=job.tenant_id,
        status=job.status,
        model_type=job.model_type,
        num_rows=job.num_rows,
        output_format=job.output_format,
        fidelity_score=job.fidelity_score,
        privacy_score=job.privacy_score,
        output_uri=job.output_uri,
        training_time_s=job.training_time_s,
        generation_time_s=job.generation_time_s,
        error_message=job.error_message,
    )


@router.get(
    "/tabular/generate",
    response_model=JobListResponse,
    summary="List generation jobs",
    description="List all generation jobs for the current tenant with pagination.",
)
async def list_generation_jobs(
    page: int = 1,
    page_size: int = 20,
    status_filter: str | None = None,
    x_tenant_id: uuid.UUID | None = None,
    service: GenerationService = Depends(get_generation_service),
) -> JobListResponse:
    """List generation jobs with pagination and optional status filter.

    Args:
        page: 1-based page number (default 1).
        page_size: Results per page (default 20, max 100).
        status_filter: Optional status to filter by.
        x_tenant_id: Tenant UUID from JWT/header.
        service: Generation service dependency.

    Returns:
        Paginated JobListResponse.
    """
    tenant_id = x_tenant_id or uuid.uuid4()
    jobs, total = await service.list_jobs(
        tenant_id=tenant_id,
        page=page,
        page_size=min(page_size, 100),
        status=status_filter,
    )

    return JobListResponse(
        items=[
            GenerationJobResponse(
                job_id=job.id,
                tenant_id=job.tenant_id,
                status=job.status,
                model_type=job.model_type,
                num_rows=job.num_rows,
                output_format=job.output_format,
                fidelity_score=job.fidelity_score,
                privacy_score=job.privacy_score,
                output_uri=job.output_uri,
                training_time_s=job.training_time_s,
                generation_time_s=job.generation_time_s,
                error_message=job.error_message,
            )
            for job in jobs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/tabular/generate/{job_id}",
    response_model=GenerationJobResponse,
    summary="Get generation job status",
    description="Poll for generation job status, progress, and quality scores.",
)
async def get_generation_job(
    job_id: uuid.UUID,
    x_tenant_id: uuid.UUID | None = None,
    service: GenerationService = Depends(get_generation_service),
) -> GenerationJobResponse:
    """Get the current status of a generation job.

    Args:
        job_id: Generation job UUID.
        x_tenant_id: Tenant UUID from JWT/header.
        service: Generation service dependency.

    Returns:
        GenerationJobResponse with current status and scores.

    Raises:
        HTTPException 404: If job not found.
    """
    tenant_id = x_tenant_id or uuid.uuid4()

    try:
        job = await service.get_job(job_id, tenant_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return GenerationJobResponse(
        job_id=job.id,
        tenant_id=job.tenant_id,
        status=job.status,
        model_type=job.model_type,
        num_rows=job.num_rows,
        output_format=job.output_format,
        fidelity_score=job.fidelity_score,
        privacy_score=job.privacy_score,
        output_uri=job.output_uri,
        training_time_s=job.training_time_s,
        generation_time_s=job.generation_time_s,
        error_message=job.error_message,
    )


@router.get(
    "/tabular/generate/{job_id}/download",
    response_model=DownloadUrlResponse,
    summary="Get download URL for generated data",
    description="Generate a presigned 1-hour download URL for a completed job's artifact.",
)
async def get_download_url(
    job_id: uuid.UUID,
    x_tenant_id: uuid.UUID | None = None,
    service: GenerationService = Depends(get_generation_service),
) -> DownloadUrlResponse:
    """Generate a presigned download URL for a completed generation job.

    Args:
        job_id: Completed generation job UUID.
        x_tenant_id: Tenant UUID from JWT/header.
        service: Generation service dependency.

    Returns:
        DownloadUrlResponse with presigned URL.

    Raises:
        HTTPException 404: If job not found.
        HTTPException 409: If job is not in 'complete' status.
    """
    tenant_id = x_tenant_id or uuid.uuid4()

    try:
        download_url = await service.get_download_url(job_id, tenant_id)
        job = await service.get_job(job_id, tenant_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return DownloadUrlResponse(
        job_id=job_id,
        download_url=download_url,
        filename=f"output.{job.output_format}",
        expires_in_seconds=3600,
    )


# ---------------------------------------------------------------------------
# Profile endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/tabular/profiles",
    response_model=ProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a generation profile",
    description="Create a reusable configuration profile for synthetic data generation.",
)
async def create_profile(
    request: ProfileCreate,
    x_tenant_id: uuid.UUID | None = None,
    service: ProfileService = Depends(get_profile_service),
) -> ProfileResponse:
    """Create a new generation profile.

    Args:
        request: Profile creation request.
        x_tenant_id: Tenant UUID from JWT/header.
        service: Profile service dependency.

    Returns:
        ProfileResponse with newly created profile.

    Raises:
        HTTPException 409: If a profile with the same name already exists.
    """
    tenant_id = x_tenant_id or uuid.uuid4()

    try:
        profile = await service.create_profile(
            tenant_id=tenant_id,
            name=request.name,
            model_type=request.model_type,
            description=request.description,
            default_dp_epsilon=request.default_dp_epsilon,
            default_dp_delta=request.default_dp_delta,
            default_constraints=[c.model_dump() for c in request.default_constraints],
            column_mappings=request.column_mappings,
        )
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return ProfileResponse.model_validate(profile)


@router.get(
    "/tabular/profiles",
    response_model=list[ProfileResponse],
    summary="List generation profiles",
    description="List all active generation profiles for the current tenant.",
)
async def list_profiles(
    active_only: bool = True,
    x_tenant_id: uuid.UUID | None = None,
    service: ProfileService = Depends(get_profile_service),
) -> list[ProfileResponse]:
    """List generation profiles for the current tenant.

    Args:
        active_only: If True, only return active profiles.
        x_tenant_id: Tenant UUID from JWT/header.
        service: Profile service dependency.

    Returns:
        List of ProfileResponse objects.
    """
    tenant_id = x_tenant_id or uuid.uuid4()
    profiles = await service.list_profiles(tenant_id, active_only)
    return [ProfileResponse.model_validate(p) for p in profiles]


@router.put(
    "/tabular/profiles/{profile_id}",
    response_model=ProfileResponse,
    summary="Update a generation profile",
    description="Apply partial updates to a generation profile.",
)
async def update_profile(
    profile_id: uuid.UUID,
    request: ProfileUpdate,
    x_tenant_id: uuid.UUID | None = None,
    service: ProfileService = Depends(get_profile_service),
) -> ProfileResponse:
    """Update a generation profile.

    Args:
        profile_id: Profile UUID to update.
        request: Partial update request (only provided fields are changed).
        x_tenant_id: Tenant UUID from JWT/header.
        service: Profile service dependency.

    Returns:
        Updated ProfileResponse.

    Raises:
        HTTPException 404: If profile not found.
    """
    tenant_id = x_tenant_id or uuid.uuid4()

    updates = request.model_dump(exclude_none=True)
    if "default_constraints" in updates:
        updates["default_constraints"] = [
            c.model_dump() if hasattr(c, "model_dump") else c
            for c in (request.default_constraints or [])
        ]

    try:
        profile = await service.update_profile(profile_id, tenant_id, updates)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return ProfileResponse.model_validate(profile)


# ---------------------------------------------------------------------------
# Multi-table endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/tabular/multi-table/analyze",
    response_model=MultiTableSchemaResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyze a multi-table schema",
    description="Register and analyze a multi-table schema with FK relationships.",
)
async def analyze_multi_table_schema(
    request: MultiTableAnalysisRequest,
    x_tenant_id: uuid.UUID | None = None,
    service: MultiTableService = Depends(get_multi_table_service),
) -> MultiTableSchemaResponse:
    """Register a multi-table schema for analysis and future synthesis.

    Args:
        request: Multi-table analysis request with table URIs and relationships.
        x_tenant_id: Tenant UUID from JWT/header.
        service: Multi-table service dependency.

    Returns:
        MultiTableSchemaResponse with registered schema details.

    Raises:
        HTTPException 409: If schema name already exists.
    """
    tenant_id = x_tenant_id or uuid.uuid4()

    relationships_raw = [r.model_dump() for r in request.relationships]

    try:
        schema = await service.register_schema(
            tenant_id=tenant_id,
            name=request.name,
            tables={},  # populated by async analysis task
            relationships=relationships_raw,
            source_uris=request.source_uris,
            description=request.description,
        )
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return MultiTableSchemaResponse(
        id=schema.id,
        tenant_id=schema.tenant_id,
        name=schema.name,
        description=schema.description,
        table_count=len(request.source_uris),
        relationship_count=len(request.relationships),
        total_generation_jobs=schema.total_generation_jobs,
    )


@router.post(
    "/tabular/multi-table/generate",
    response_model=GenerationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate linked synthetic tables",
    description="Generate a complete set of FK-linked synthetic tables using HMA.",
)
async def generate_multi_table(
    request: MultiTableGenerateRequest,
    x_tenant_id: uuid.UUID | None = None,
    service: MultiTableService = Depends(get_multi_table_service),
    generation_service: GenerationService = Depends(get_generation_service),
) -> GenerationJobResponse:
    """Generate linked synthetic tables preserving FK relationships.

    Args:
        request: Multi-table generation request.
        x_tenant_id: Tenant UUID from JWT/header.
        service: Multi-table service dependency.
        generation_service: Generation service for job creation.

    Returns:
        GenerationJobResponse for the queued multi-table job.

    Raises:
        HTTPException 404: If schema not found.
    """
    tenant_id = x_tenant_id or uuid.uuid4()

    try:
        schema = await service.get_schema(request.schema_id, tenant_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    total_rows = sum(request.num_rows_per_table.values())

    try:
        job = await generation_service.create_job(
            tenant_id=tenant_id,
            num_rows=total_rows,
            model_type=request.model,
            source_schema={
                "multi_table_schema_id": str(schema.id),
                "tables": schema.tables,
                "relationships": schema.relationships,
                "num_rows_per_table": request.num_rows_per_table,
            },
            output_format=request.output_format,
        )
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return GenerationJobResponse(
        job_id=job.id,
        tenant_id=job.tenant_id,
        status=job.status,
        model_type=job.model_type,
        num_rows=job.num_rows,
        output_format=job.output_format,
        fidelity_score=job.fidelity_score,
        privacy_score=job.privacy_score,
        output_uri=job.output_uri,
        training_time_s=job.training_time_s,
        generation_time_s=job.generation_time_s,
        error_message=job.error_message,
    )

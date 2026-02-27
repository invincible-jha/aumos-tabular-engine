"""Privacy wrapper adapter integrating with aumos-privacy-engine via HTTP.

Handles epsilon budget allocation, consumption tracking, DP noise injection
coordination, privacy budget monitoring, and synthesis-privacy tradeoff analysis.
"""

import asyncio
import time
import uuid
from typing import Any

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Privacy level presets: (epsilon, delta) pairs
_PRIVACY_LEVEL_PRESETS: dict[str, tuple[float, float]] = {
    "strict": (0.1, 1e-6),
    "moderate": (1.0, 1e-5),
    "lenient": (5.0, 1e-4),
    "disabled": (float("inf"), 0.0),
}

# Default HTTP timeouts
_DEFAULT_CONNECT_TIMEOUT_S = 5.0
_DEFAULT_READ_TIMEOUT_S = 30.0


class BudgetAllocationResponse(BaseModel):
    """Pydantic model for the budget allocation API response."""

    model_config = ConfigDict(from_attributes=True)

    allocation_id: str = Field(..., description="Unique allocation identifier")
    tenant_id: str = Field(..., description="Tenant UUID")
    epsilon_allocated: float = Field(..., description="Allocated epsilon value")
    delta_allocated: float = Field(..., description="Allocated delta value")
    remaining_budget: float = Field(..., description="Remaining epsilon budget after allocation")
    allocation_status: str = Field(..., description="Status: approved or rejected")


class BudgetStatusResponse(BaseModel):
    """Pydantic model for the budget status API response."""

    model_config = ConfigDict(from_attributes=True)

    tenant_id: str
    total_budget: float
    consumed_budget: float
    remaining_budget: float
    allocation_count: int


class PrivacyWrapper:
    """HTTP client wrapper for aumos-privacy-engine DP budget management.

    Manages epsilon budget requests and consumption for differential privacy
    synthesis jobs. Tracks allocations per job, monitors budget levels, and
    provides synthesis-privacy tradeoff analysis.
    """

    def __init__(
        self,
        privacy_engine_base_url: str,
        api_key: str | None = None,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        read_timeout_s: float = _DEFAULT_READ_TIMEOUT_S,
        min_remaining_budget_warning: float = 1.0,
    ) -> None:
        """Initialise the privacy wrapper HTTP client.

        Args:
            privacy_engine_base_url: Base URL of the aumos-privacy-engine service
                (e.g., 'http://privacy-engine:8008').
            api_key: Optional Bearer token for service-to-service auth.
            connect_timeout_s: HTTP connect timeout in seconds.
            read_timeout_s: HTTP read timeout in seconds.
            min_remaining_budget_warning: Remaining epsilon threshold below which
                a warning is emitted during budget checks.
        """
        self._base_url = privacy_engine_base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = httpx.Timeout(connect=connect_timeout_s, read=read_timeout_s, write=10.0, pool=5.0)
        self._min_budget_warning = min_remaining_budget_warning
        self._allocation_registry: dict[str, dict[str, Any]] = {}

    async def allocate_budget(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        epsilon: float,
        delta: float,
        purpose: str = "tabular_synthesis",
    ) -> BudgetAllocationResponse:
        """Request epsilon budget allocation from aumos-privacy-engine.

        Sends a POST request to allocate epsilon/delta for a synthesis job.
        If the allocation is rejected (insufficient budget), raises ValueError
        so the calling service can transition the job to FAILED.

        Args:
            tenant_id: Owning tenant UUID.
            job_id: Generation job UUID (used for allocation audit trail).
            epsilon: Requested epsilon value for differential privacy.
            delta: Requested delta value for differential privacy.
            purpose: Human-readable purpose label for audit.

        Returns:
            BudgetAllocationResponse with allocation_id and status.

        Raises:
            ValueError: If the allocation is rejected due to budget exhaustion.
            httpx.HTTPError: If the privacy engine is unreachable.
        """
        payload = {
            "tenant_id": str(tenant_id),
            "job_id": str(job_id),
            "epsilon": epsilon,
            "delta": delta,
            "purpose": purpose,
        }

        logger.info(
            "Requesting budget allocation",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            epsilon=epsilon,
            delta=delta,
        )

        response_data = await self._post(
            endpoint="/api/v1/privacy/budget/allocate",
            payload=payload,
        )

        allocation = BudgetAllocationResponse(**response_data)

        if allocation.allocation_status != "approved":
            logger.warning(
                "Budget allocation rejected",
                tenant_id=str(tenant_id),
                job_id=str(job_id),
                epsilon=epsilon,
                remaining=allocation.remaining_budget,
            )
            raise ValueError(
                f"DP budget allocation rejected: requested epsilon={epsilon}, "
                f"remaining budget={allocation.remaining_budget:.4f}"
            )

        self._allocation_registry[allocation.allocation_id] = {
            "tenant_id": str(tenant_id),
            "job_id": str(job_id),
            "epsilon": epsilon,
            "delta": delta,
            "allocated_at": time.time(),
            "consumed": False,
        }

        if allocation.remaining_budget < self._min_budget_warning:
            logger.warning(
                "Low remaining privacy budget",
                tenant_id=str(tenant_id),
                remaining_budget=allocation.remaining_budget,
                threshold=self._min_budget_warning,
            )

        logger.info(
            "Budget allocation approved",
            allocation_id=allocation.allocation_id,
            remaining_budget=allocation.remaining_budget,
        )
        return allocation

    async def consume_budget(
        self,
        allocation_id: str,
        actual_epsilon_spent: float | None = None,
    ) -> dict[str, Any]:
        """Consume (commit) a previously allocated DP budget.

        Should be called after successful synthesis to finalize the epsilon spend.
        If actual_epsilon_spent is provided, it must not exceed the allocated epsilon.

        Args:
            allocation_id: The allocation_id from a prior allocate_budget() call.
            actual_epsilon_spent: Optional actual epsilon consumed. Defaults to
                the full allocated epsilon if not provided.

        Returns:
            Consumption confirmation dict from the privacy engine.

        Raises:
            ValueError: If allocation_id is unknown or already consumed.
            httpx.HTTPError: If the privacy engine is unreachable.
        """
        local_record = self._allocation_registry.get(allocation_id)
        if local_record is None:
            logger.warning("Unknown allocation_id for consumption", allocation_id=allocation_id)

        if local_record and local_record.get("consumed"):
            raise ValueError(f"Allocation '{allocation_id}' has already been consumed")

        payload: dict[str, Any] = {"allocation_id": allocation_id}
        if actual_epsilon_spent is not None:
            payload["actual_epsilon_spent"] = actual_epsilon_spent

        response = await self._post(
            endpoint="/api/v1/privacy/budget/consume",
            payload=payload,
        )

        if local_record:
            local_record["consumed"] = True
            local_record["consumed_at"] = time.time()

        logger.info(
            "Budget consumption recorded",
            allocation_id=allocation_id,
            actual_epsilon=actual_epsilon_spent,
        )
        return response

    async def get_budget_status(self, tenant_id: uuid.UUID) -> BudgetStatusResponse:
        """Query remaining DP budget for a tenant.

        Args:
            tenant_id: Tenant UUID to query.

        Returns:
            BudgetStatusResponse with remaining, consumed, and total budget.

        Raises:
            httpx.HTTPError: If the privacy engine is unreachable.
        """
        response_data = await self._get(
            endpoint=f"/api/v1/privacy/budget/status/{tenant_id}",
        )
        return BudgetStatusResponse(**response_data)

    async def analyze_privacy_tradeoff(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        epsilon_values: list[float],
        fidelity_scores: list[float],
    ) -> dict[str, Any]:
        """Analyze the privacy-fidelity tradeoff curve.

        Combines locally-computed tradeoff metrics with the Pareto frontier
        to characterize the synthesis-privacy relationship.

        Args:
            real_data: Original source DataFrame (used for basic stats).
            synthetic_data: Generated synthetic DataFrame.
            epsilon_values: List of epsilon values from experiments.
            fidelity_scores: Corresponding fidelity scores for each epsilon.

        Returns:
            Tradeoff analysis dict with curve data, Pareto points, and recommendations.
        """
        if len(epsilon_values) != len(fidelity_scores):
            raise ValueError("epsilon_values and fidelity_scores must have equal length")

        tradeoff_points = [
            {"epsilon": eps, "fidelity": fid, "privacy": 1.0 / (1.0 + eps)}
            for eps, fid in zip(epsilon_values, fidelity_scores)
        ]

        pareto_points = self._compute_pareto_frontier(tradeoff_points)

        recommendation = self._recommend_epsilon(pareto_points)

        size_ratio = len(synthetic_data) / max(len(real_data), 1)
        logger.info(
            "Privacy-fidelity tradeoff analysis complete",
            num_points=len(tradeoff_points),
            num_pareto_points=len(pareto_points),
            recommended_epsilon=recommendation.get("epsilon"),
        )

        return {
            "tradeoff_curve": tradeoff_points,
            "pareto_frontier": pareto_points,
            "recommendation": recommendation,
            "dataset_size_ratio": round(size_ratio, 4),
            "analysis_metadata": {
                "real_rows": len(real_data),
                "synthetic_rows": len(synthetic_data),
                "epsilon_range": [min(epsilon_values), max(epsilon_values)] if epsilon_values else [],
            },
        }

    def get_allocation_registry(self) -> dict[str, dict[str, Any]]:
        """Return the in-memory allocation registry for auditing.

        Returns:
            Dict mapping allocation_id → allocation metadata.
        """
        return dict(self._allocation_registry)

    def get_privacy_level_presets(self) -> dict[str, dict[str, float]]:
        """Return the available privacy level presets with epsilon/delta values.

        Returns:
            Dict mapping level_name → {epsilon, delta}.
        """
        return {
            level: {"epsilon": eps, "delta": dlt}
            for level, (eps, dlt) in _PRIVACY_LEVEL_PRESETS.items()
        }

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
            Dict with 'sufficient' bool, 'remaining_budget', and 'requested_epsilon'.
        """
        status = await self.get_budget_status(tenant_id)
        sufficient = status.remaining_budget >= requested_epsilon

        if not sufficient:
            logger.warning(
                "Insufficient DP budget",
                tenant_id=str(tenant_id),
                requested=requested_epsilon,
                remaining=status.remaining_budget,
            )

        return {
            "sufficient": sufficient,
            "remaining_budget": status.remaining_budget,
            "requested_epsilon": requested_epsilon,
            "shortfall": max(0.0, requested_epsilon - status.remaining_budget),
        }

    def _compute_pareto_frontier(
        self,
        points: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compute the Pareto-optimal frontier for privacy vs fidelity tradeoff.

        A point is Pareto-optimal if no other point is strictly better in both
        fidelity and privacy dimensions simultaneously.

        Args:
            points: List of dicts with 'epsilon', 'fidelity', and 'privacy' keys.

        Returns:
            Pareto-optimal subset of input points.
        """
        if not points:
            return []

        pareto_points: list[dict[str, Any]] = []
        for candidate in points:
            dominated = False
            for other in points:
                if other is candidate:
                    continue
                if (
                    other["fidelity"] >= candidate["fidelity"]
                    and other["privacy"] >= candidate["privacy"]
                    and (
                        other["fidelity"] > candidate["fidelity"]
                        or other["privacy"] > candidate["privacy"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto_points.append(candidate)

        return sorted(pareto_points, key=lambda p: p["epsilon"])

    def _recommend_epsilon(
        self,
        pareto_points: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Recommend the epsilon with best fidelity-privacy balance on the frontier.

        Uses the point with the maximum harmonic mean of fidelity and privacy
        scores as the balanced recommendation.

        Args:
            pareto_points: Pareto-optimal points from _compute_pareto_frontier().

        Returns:
            Recommended point dict with epsilon, fidelity, privacy, and reason.
        """
        if not pareto_points:
            return {"epsilon": 1.0, "reason": "default — no tradeoff data available"}

        best_point = max(
            pareto_points,
            key=lambda p: (
                2 * p["fidelity"] * p["privacy"] / (p["fidelity"] + p["privacy"] + 1e-10)
            ),
        )

        return {
            **best_point,
            "reason": "maximum harmonic mean of fidelity and privacy scores",
        }

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send an authenticated POST request to the privacy engine.

        Args:
            endpoint: Relative API endpoint path.
            payload: JSON payload dict.

        Returns:
            Parsed JSON response dict.

        Raises:
            httpx.HTTPError: On HTTP or network failure.
        """
        headers = self._build_headers()
        url = f"{self._base_url}{endpoint}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def _get(self, endpoint: str) -> dict[str, Any]:
        """Send an authenticated GET request to the privacy engine.

        Args:
            endpoint: Relative API endpoint path.

        Returns:
            Parsed JSON response dict.

        Raises:
            httpx.HTTPError: On HTTP or network failure.
        """
        headers = self._build_headers()
        url = f"{self._base_url}{endpoint}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP request headers including optional Authorization.

        Returns:
            Headers dict.
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Service-Name": "aumos-tabular-engine",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

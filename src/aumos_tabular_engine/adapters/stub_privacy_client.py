"""Stub privacy client for demo mode.

Always approves budget allocations without contacting aumos-privacy-engine.
This is intentional — demo mode is for functionality demonstration, not
privacy testing. Production deployments must use the real privacy client.
"""

from __future__ import annotations

import uuid

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class StubPrivacyClient:
    """No-op privacy client for demo mode.

    Returns fake allocation IDs and silently no-ops on consume.
    Never use this in production — it provides no privacy guarantees.
    """

    async def allocate_budget(
        self,
        tenant_id: uuid.UUID,
        epsilon: float,
        delta: float,
        purpose: str = "tabular_synthesis",
    ) -> str:
        """Return a fake allocation ID without contacting the privacy engine.

        Args:
            tenant_id: Owning tenant UUID.
            epsilon: Requested epsilon (ignored in demo mode).
            delta: Requested delta (ignored in demo mode).
            purpose: Purpose label (ignored in demo mode).

        Returns:
            Fake allocation ID string for demo tracking.
        """
        allocation_id = f"demo-allocation-{uuid.uuid4()}"
        logger.info(
            "stub_privacy_client_allocate",
            tenant_id=str(tenant_id),
            epsilon=epsilon,
            delta=delta,
            allocation_id=allocation_id,
        )
        return allocation_id

    async def consume_budget(self, allocation_id: str) -> None:
        """No-op consumption in demo mode.

        Args:
            allocation_id: Allocation ID to consume (ignored in demo mode).
        """
        logger.info(
            "stub_privacy_client_consume",
            allocation_id=allocation_id,
        )

    async def get_budget_status(self, tenant_id: uuid.UUID) -> dict[str, float]:
        """Return fake unlimited budget status.

        Args:
            tenant_id: Tenant UUID.

        Returns:
            Dict with unlimited budget values for demo display.
        """
        return {
            "remaining_epsilon": 999.0,
            "consumed_epsilon": 0.0,
            "total_epsilon": 999.0,
        }

    async def validate_epsilon_budget(
        self,
        tenant_id: uuid.UUID,
        requested_epsilon: float,
    ) -> dict[str, object]:
        """Always approve in demo mode.

        Args:
            tenant_id: Tenant UUID.
            requested_epsilon: Requested epsilon.

        Returns:
            Dict with sufficient=True and unlimited budget values.
        """
        return {
            "sufficient": True,
            "remaining": 999.0,
            "requested": requested_epsilon,
        }

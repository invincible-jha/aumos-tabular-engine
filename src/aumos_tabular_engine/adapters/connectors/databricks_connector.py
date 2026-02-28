"""Databricks source connector for tabular engine data ingestion.

Fetches query results from Databricks SQL Warehouse and stages to MinIO as Parquet.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from typing import Any

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class DatabricksConnector:
    """Fetches data from Databricks SQL Warehouse and stages to MinIO.

    Args:
        server_hostname: Databricks workspace hostname.
        http_path: SQL warehouse HTTP path.
        access_token: Personal access token or OAuth token.
            In production, resolve via aumos-secrets-vault.
        catalog: Unity Catalog name (optional).
        schema: Schema/database name (optional).
    """

    def __init__(
        self,
        server_hostname: str,
        http_path: str,
        access_token: str,
        catalog: str | None = None,
        schema: str | None = None,
    ) -> None:
        self._server_hostname = server_hostname
        self._http_path = http_path
        self._access_token = access_token
        self._catalog = catalog
        self._schema = schema

    async def fetch_query(
        self,
        query: str,
        staging_bucket: str,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        storage_upload_fn: Any,
    ) -> str:
        """Execute query, write result as Parquet to MinIO, return storage URI.

        Args:
            query: SQL query to execute against Databricks.
            staging_bucket: MinIO bucket for staging.
            tenant_id: Owning tenant UUID.
            job_id: Generation job UUID.
            storage_upload_fn: Async callable(data: bytes, key: str) -> str.

        Returns:
            MinIO URI of the staged Parquet file.

        Raises:
            ImportError: If databricks-sql-connector is not installed.
        """
        try:
            from databricks import sql as databricks_sql
        except ImportError as exc:
            raise ImportError(
                "databricks-sql-connector is required. "
                "Install with: pip install databricks-sql-connector>=3.3"
            ) from exc

        logger.info(
            "databricks_fetch_start",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            server_hostname=self._server_hostname,
        )

        def _execute() -> pd.DataFrame:
            with databricks_sql.connect(
                server_hostname=self._server_hostname,
                http_path=self._http_path,
                access_token=self._access_token,
                catalog=self._catalog,
                schema=self._schema,
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    return cursor.fetchall_arrow().to_pandas()

        df = await asyncio.to_thread(_execute)

        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        key = f"tab-staging/{tenant_id}/{job_id}/source.parquet"
        uri: str = await storage_upload_fn(buf.getvalue(), key)

        logger.info(
            "databricks_fetch_complete",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            rows=len(df),
        )
        return uri

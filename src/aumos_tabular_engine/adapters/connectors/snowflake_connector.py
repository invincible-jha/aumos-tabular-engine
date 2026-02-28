"""Snowflake source connector for tabular engine data ingestion.

Fetches query results from Snowflake and stages to MinIO as Parquet,
returning a MinIO URI for the tabular engine to consume.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from typing import Any

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class SnowflakeConnector:
    """Fetches data from Snowflake and stages to MinIO.

    Executes a parameterized SQL query against Snowflake, retrieves the
    result as an Arrow table, converts to pandas, serializes as Parquet,
    and uploads to MinIO staging.

    Args:
        account: Snowflake account identifier (e.g., "myorg-myaccount").
        warehouse: Virtual warehouse name.
        database: Database name.
        schema: Schema name.
        user: Snowflake username.
        password: Snowflake password (prefer secrets-vault credentials in production).
    """

    def __init__(
        self,
        account: str,
        warehouse: str,
        database: str,
        schema: str,
        user: str,
        password: str,
    ) -> None:
        self._account = account
        self._warehouse = warehouse
        self._database = database
        self._schema = schema
        self._user = user
        self._password = password

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
            query: Parameterized SQL query to execute. Must not contain user-provided
                literals — use bind parameters via snowflake-connector.
            staging_bucket: MinIO bucket for staging.
            tenant_id: Owning tenant UUID (for staging path isolation).
            job_id: Generation job UUID for unique key generation.
            storage_upload_fn: Async callable(data: bytes, key: str) -> str.

        Returns:
            MinIO URI of the staged Parquet file.

        Raises:
            ImportError: If snowflake-connector-python is not installed.
            RuntimeError: If the Snowflake query fails.
        """
        try:
            import snowflake.connector
        except ImportError as exc:
            raise ImportError(
                "snowflake-connector-python is required. "
                "Install with: pip install 'snowflake-connector-python[pandas,arrow]>=3.7'"
            ) from exc

        logger.info(
            "snowflake_fetch_start",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            account=self._account,
            database=self._database,
        )

        def _execute() -> pd.DataFrame:
            conn = snowflake.connector.connect(
                account=self._account,
                user=self._user,
                password=self._password,
                warehouse=self._warehouse,
                database=self._database,
                schema=self._schema,
            )
            try:
                cursor = conn.cursor()
                cursor.execute(query)
                arrow_result = cursor.fetch_arrow_all()
                return arrow_result.to_pandas()
            finally:
                conn.close()

        df = await asyncio.to_thread(_execute)

        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        key = f"tab-staging/{tenant_id}/{job_id}/source.parquet"
        uri: str = await storage_upload_fn(buf.getvalue(), key)

        logger.info(
            "snowflake_fetch_complete",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            rows=len(df),
            uri=uri,
        )
        return uri

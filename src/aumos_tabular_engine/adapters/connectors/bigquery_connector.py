"""BigQuery source connector for tabular engine data ingestion.

Fetches query results from Google BigQuery and stages to MinIO as Parquet.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from typing import Any

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class BigQueryConnector:
    """Fetches data from Google BigQuery and stages to MinIO.

    Args:
        project_id: Google Cloud project ID.
        credentials_json: Service account credentials JSON string.
            In production, resolve via aumos-secrets-vault.
        location: BigQuery dataset location (default: "US").
    """

    def __init__(
        self,
        project_id: str,
        credentials_json: str | None = None,
        location: str = "US",
    ) -> None:
        self._project_id = project_id
        self._credentials_json = credentials_json
        self._location = location

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
            query: Standard SQL query against BigQuery.
            staging_bucket: MinIO bucket for staging.
            tenant_id: Owning tenant UUID.
            job_id: Generation job UUID.
            storage_upload_fn: Async callable(data: bytes, key: str) -> str.

        Returns:
            MinIO URI of the staged Parquet file.

        Raises:
            ImportError: If google-cloud-bigquery is not installed.
        """
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError as exc:
            raise ImportError(
                "google-cloud-bigquery is required. "
                "Install with: pip install 'google-cloud-bigquery[pandas]>=3.19'"
            ) from exc

        logger.info(
            "bigquery_fetch_start",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            project_id=self._project_id,
        )

        def _execute() -> pd.DataFrame:
            if self._credentials_json:
                import json
                creds_dict = json.loads(self._credentials_json)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                client = bigquery.Client(
                    project=self._project_id,
                    credentials=credentials,
                    location=self._location,
                )
            else:
                client = bigquery.Client(
                    project=self._project_id,
                    location=self._location,
                )
            return client.query(query).to_dataframe()

        df = await asyncio.to_thread(_execute)

        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        key = f"tab-staging/{tenant_id}/{job_id}/source.parquet"
        uri: str = await storage_upload_fn(buf.getvalue(), key)

        logger.info(
            "bigquery_fetch_complete",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            rows=len(df),
        )
        return uri

"""Local filesystem storage adapter for demo and development mode.

Provides the same interface as MinIOStorageAdapter but writes files to
a local directory. Used when AUMOS_TAB_DEMO_MODE=true or AUMOS_TAB_STORAGE_BACKEND=local.
"""

from __future__ import annotations

import pathlib
import uuid

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class LocalStorageAdapter:
    """File-system backed storage for demo mode and local development.

    Implements the same upload/download/presigned_url interface as the MinIO
    adapter so service code is unchanged regardless of the storage backend.

    Args:
        base_path: Root directory for file storage (default: ./demo-output).
    """

    def __init__(self, base_path: str = "./demo-output") -> None:
        self._base_path = pathlib.Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    async def upload(self, data: bytes, key: str) -> str:
        """Write bytes to local filesystem and return a file:// URI.

        Args:
            data: Raw bytes to write.
            key: Relative path key (e.g., 'tab-jobs/{job_id}/output.parquet').

        Returns:
            file:// URI pointing to the written file.
        """
        path = self._base_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        uri = f"file://{path.resolve()}"
        logger.debug("local_storage_upload", key=key, bytes=len(data), uri=uri)
        return uri

    async def download(self, key: str) -> bytes:
        """Read bytes from local filesystem.

        Args:
            key: Relative path key.

        Returns:
            Raw bytes of the stored file.

        Raises:
            FileNotFoundError: If the key does not exist.
        """
        path = self._base_path / key
        if not path.exists():
            raise FileNotFoundError(f"Local storage key not found: {key}")
        return path.read_bytes()

    async def get_presigned_url(
        self,
        key: str,
        expires_seconds: int = 3600,
    ) -> str:
        """Return a file:// URL for the given key.

        In demo mode, the 'presigned URL' is simply a local file path.
        In production, use MinIOStorageAdapter for real presigned URLs.

        Args:
            key: Relative path key.
            expires_seconds: Ignored in local mode (no expiry for local files).

        Returns:
            file:// URI pointing to the stored file.
        """
        path = self._base_path / key
        return f"file://{path.resolve()}"

    async def upload_to_tenant_bucket(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        data: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload artifact to tenant-namespaced path.

        Args:
            tenant_id: Owning tenant UUID.
            job_id: Generation job UUID.
            data: Raw artifact bytes.
            filename: Artifact filename.
            content_type: MIME type (ignored in local mode).

        Returns:
            file:// URI for the stored artifact.
        """
        key = f"tab-jobs/{tenant_id}/{job_id}/{filename}"
        return await self.upload(data, key)

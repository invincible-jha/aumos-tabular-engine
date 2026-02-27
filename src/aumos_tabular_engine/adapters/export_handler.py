"""Export handler adapter for serializing and uploading synthetic datasets.

Handles CSV, Parquet, and Excel exports with configurable compression and
encoding. Supports chunked export for large DataFrames and generates export
metadata manifests. Uploads artifacts to MinIO/S3-compatible storage.
"""

import asyncio
import io
import uuid
from functools import partial
from typing import Any, Literal

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Supported export formats and their MIME types
_FORMAT_MIME_TYPES: dict[str, str] = {
    "csv": "text/csv",
    "parquet": "application/octet-stream",
    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "json": "application/json",
    "jsonl": "application/x-ndjson",
}

# Parquet compression options
_PARQUET_COMPRESSION_OPTIONS = frozenset({"snappy", "gzip", "zstd", "brotli", "lz4", "none"})

# Default chunk size for chunked export (rows per chunk)
_DEFAULT_CHUNK_SIZE = 100_000

ExportFormat = Literal["csv", "parquet", "excel", "json", "jsonl"]
ParquetCompression = Literal["snappy", "gzip", "zstd", "brotli", "lz4", "none"]


class ExportHandler:
    """Serialize synthetic DataFrames to multiple file formats and upload to storage.

    Supports CSV (configurable delimiter/encoding), Parquet (snappy/gzip/zstd),
    Excel (.xlsx), and JSON/JSONL. Handles chunked export for large datasets and
    generates JSON metadata manifests alongside each export.
    """

    def __init__(
        self,
        storage_client: Any | None = None,
        default_parquet_compression: ParquetCompression = "snappy",
        default_csv_delimiter: str = ",",
        default_csv_encoding: str = "utf-8",
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        export_base_path: str = "tab-exports",
    ) -> None:
        """Initialise the export handler.

        Args:
            storage_client: MinIO/S3 storage adapter implementing StorageProtocol.
                If None, export methods return raw bytes without uploading.
            default_parquet_compression: Default Parquet compression codec.
            default_csv_delimiter: Default CSV field delimiter character.
            default_csv_encoding: Default CSV character encoding.
            chunk_size: Number of rows per chunk during chunked exports.
            export_base_path: Base storage path prefix for exported files.
        """
        self._storage = storage_client
        self._parquet_compression = default_parquet_compression
        self._csv_delimiter = default_csv_delimiter
        self._csv_encoding = default_csv_encoding
        self._chunk_size = chunk_size
        self._export_base_path = export_base_path

    async def export(
        self,
        data: pd.DataFrame,
        output_format: ExportFormat,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        filename: str | None = None,
        **format_kwargs: Any,
    ) -> dict[str, Any]:
        """Export a DataFrame in the specified format and upload to storage.

        Args:
            data: Synthetic DataFrame to export.
            output_format: Target format: csv, parquet, excel, json, jsonl.
            tenant_id: Owning tenant for storage path isolation.
            job_id: Generation job UUID for object key namespacing.
            filename: Override filename (defaults to 'output.<format>').
            **format_kwargs: Format-specific options (delimiter, compression, etc.).

        Returns:
            Export result dict with filename, storage_uri, bytes_written,
            row_count, column_count, and content_type.
        """
        if output_format not in _FORMAT_MIME_TYPES:
            raise ValueError(
                f"Unsupported export format '{output_format}'. "
                f"Supported: {list(_FORMAT_MIME_TYPES.keys())}"
            )

        effective_filename = filename or _default_filename(output_format)
        content_type = _FORMAT_MIME_TYPES[output_format]

        logger.info(
            "Exporting dataset",
            output_format=output_format,
            num_rows=len(data),
            num_columns=len(data.columns),
            tenant_id=str(tenant_id),
            job_id=str(job_id),
        )

        loop = asyncio.get_running_loop()
        artifact_bytes = await loop.run_in_executor(
            None,
            partial(self._serialize, data, output_format, format_kwargs),
        )

        storage_uri: str | None = None
        if self._storage is not None:
            storage_uri = await self._storage.upload(
                tenant_id=tenant_id,
                job_id=job_id,
                data=artifact_bytes,
                filename=effective_filename,
                content_type=content_type,
            )

        metadata = await self._generate_export_metadata(
            data=data,
            output_format=output_format,
            filename=effective_filename,
            bytes_written=len(artifact_bytes),
            storage_uri=storage_uri,
        )

        if self._storage is not None:
            meta_bytes = _dict_to_json_bytes(metadata)
            await self._storage.upload(
                tenant_id=tenant_id,
                job_id=job_id,
                data=meta_bytes,
                filename=f"{effective_filename}.meta.json",
                content_type="application/json",
            )

        logger.info(
            "Export complete",
            output_format=output_format,
            bytes_written=len(artifact_bytes),
            storage_uri=storage_uri,
        )
        return metadata

    async def export_csv(
        self,
        data: pd.DataFrame,
        delimiter: str | None = None,
        encoding: str | None = None,
        include_index: bool = False,
        date_format: str | None = None,
    ) -> bytes:
        """Serialize DataFrame to CSV bytes.

        Args:
            data: DataFrame to serialize.
            delimiter: Field separator character. Defaults to instance default.
            encoding: Character encoding. Defaults to instance default.
            include_index: Whether to include row index in output.
            date_format: strftime format string for datetime columns.

        Returns:
            CSV bytes.
        """
        effective_delimiter = delimiter or self._csv_delimiter
        effective_encoding = encoding or self._csv_encoding

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                _serialize_csv,
                data,
                effective_delimiter,
                effective_encoding,
                include_index,
                date_format,
            ),
        )

    async def export_parquet(
        self,
        data: pd.DataFrame,
        compression: ParquetCompression | None = None,
        row_group_size: int | None = None,
    ) -> bytes:
        """Serialize DataFrame to Parquet bytes with configurable compression.

        Args:
            data: DataFrame to serialize.
            compression: Parquet compression codec. Defaults to instance default.
            row_group_size: Optional row group size for the Parquet file.

        Returns:
            Parquet bytes.
        """
        effective_compression = compression or self._parquet_compression
        if effective_compression == "none":
            effective_compression = None  # type: ignore[assignment]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(_serialize_parquet, data, effective_compression, row_group_size),
        )

    async def export_excel(
        self,
        data: pd.DataFrame,
        sheet_name: str = "SyntheticData",
        freeze_panes: tuple[int, int] = (1, 0),
        auto_filter: bool = True,
    ) -> bytes:
        """Serialize DataFrame to Excel (.xlsx) bytes.

        Args:
            data: DataFrame to serialize.
            sheet_name: Name of the Excel worksheet.
            freeze_panes: Row/column coordinates to freeze (default: header row).
            auto_filter: Whether to enable column auto-filtering.

        Returns:
            Excel .xlsx bytes.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(_serialize_excel, data, sheet_name, freeze_panes, auto_filter),
        )

    async def export_chunked(
        self,
        data: pd.DataFrame,
        output_format: ExportFormat,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        chunk_size: int | None = None,
        **format_kwargs: Any,
    ) -> dict[str, Any]:
        """Export large DataFrames in chunks, uploading each chunk separately.

        Splits the DataFrame into rows-per-chunk slices, serializes each slice,
        and uploads to storage as numbered chunk files. Returns a manifest of
        all uploaded chunks.

        Args:
            data: Large DataFrame to export in chunks.
            output_format: Target format for each chunk.
            tenant_id: Owning tenant UUID.
            job_id: Generation job UUID.
            chunk_size: Number of rows per chunk. Defaults to instance default.
            **format_kwargs: Format-specific serialization options.

        Returns:
            Chunked export manifest dict with chunk count, total bytes,
            and list of chunk URIs.
        """
        if self._storage is None:
            raise RuntimeError("Chunked export requires a storage client to be configured")

        effective_chunk_size = chunk_size or self._chunk_size
        total_rows = len(data)
        num_chunks = max(1, (total_rows + effective_chunk_size - 1) // effective_chunk_size)
        chunk_uris: list[str] = []
        total_bytes = 0

        logger.info(
            "Starting chunked export",
            output_format=output_format,
            total_rows=total_rows,
            num_chunks=num_chunks,
            chunk_size=effective_chunk_size,
        )

        loop = asyncio.get_running_loop()

        for chunk_index in range(num_chunks):
            start_row = chunk_index * effective_chunk_size
            end_row = min(start_row + effective_chunk_size, total_rows)
            chunk_df = data.iloc[start_row:end_row]

            chunk_bytes = await loop.run_in_executor(
                None,
                partial(self._serialize, chunk_df, output_format, format_kwargs),
            )

            extension = _format_extension(output_format)
            chunk_filename = f"chunk_{chunk_index:04d}{extension}"

            chunk_uri = await self._storage.upload(
                tenant_id=tenant_id,
                job_id=job_id,
                data=chunk_bytes,
                filename=chunk_filename,
                content_type=_FORMAT_MIME_TYPES[output_format],
            )
            chunk_uris.append(chunk_uri)
            total_bytes += len(chunk_bytes)

            logger.info(
                "Chunk uploaded",
                chunk_index=chunk_index,
                start_row=start_row,
                end_row=end_row,
                chunk_bytes=len(chunk_bytes),
            )

        manifest = {
            "export_type": "chunked",
            "output_format": output_format,
            "total_rows": total_rows,
            "num_chunks": num_chunks,
            "chunk_size": effective_chunk_size,
            "total_bytes": total_bytes,
            "chunk_uris": chunk_uris,
            "tenant_id": str(tenant_id),
            "job_id": str(job_id),
        }

        if self._storage is not None:
            manifest_bytes = _dict_to_json_bytes(manifest)
            await self._storage.upload(
                tenant_id=tenant_id,
                job_id=job_id,
                data=manifest_bytes,
                filename="chunked_export_manifest.json",
                content_type="application/json",
            )

        logger.info(
            "Chunked export complete",
            num_chunks=num_chunks,
            total_bytes=total_bytes,
        )
        return manifest

    async def _generate_export_metadata(
        self,
        data: pd.DataFrame,
        output_format: ExportFormat,
        filename: str,
        bytes_written: int,
        storage_uri: str | None,
    ) -> dict[str, Any]:
        """Generate an export metadata manifest dict.

        Args:
            data: Exported DataFrame.
            output_format: Export format used.
            filename: Output filename.
            bytes_written: Total bytes in the exported artifact.
            storage_uri: MinIO/S3 URI of the uploaded artifact.

        Returns:
            Metadata dict suitable for JSON serialization.
        """
        import datetime

        dtypes_map = {col: str(dtype) for col, dtype in data.dtypes.items()}

        return {
            "filename": filename,
            "output_format": output_format,
            "content_type": _FORMAT_MIME_TYPES[output_format],
            "row_count": len(data),
            "column_count": len(data.columns),
            "columns": list(data.columns),
            "column_dtypes": dtypes_map,
            "bytes_written": bytes_written,
            "storage_uri": storage_uri,
            "exported_at": datetime.datetime.utcnow().isoformat() + "Z",
            "null_counts": data.isna().sum().to_dict(),
        }

    def _serialize(
        self,
        data: pd.DataFrame,
        output_format: ExportFormat,
        format_kwargs: dict[str, Any],
    ) -> bytes:
        """Synchronous serialization dispatcher (runs in executor).

        Args:
            data: DataFrame to serialize.
            output_format: Target format.
            format_kwargs: Format-specific keyword arguments.

        Returns:
            Serialized bytes.
        """
        if output_format == "csv":
            return _serialize_csv(
                data,
                delimiter=format_kwargs.get("delimiter", self._csv_delimiter),
                encoding=format_kwargs.get("encoding", self._csv_encoding),
                include_index=format_kwargs.get("include_index", False),
                date_format=format_kwargs.get("date_format"),
            )
        elif output_format == "parquet":
            compression = format_kwargs.get("compression", self._parquet_compression)
            if compression == "none":
                compression = None
            return _serialize_parquet(
                data,
                compression=compression,
                row_group_size=format_kwargs.get("row_group_size"),
            )
        elif output_format == "excel":
            return _serialize_excel(
                data,
                sheet_name=format_kwargs.get("sheet_name", "SyntheticData"),
                freeze_panes=format_kwargs.get("freeze_panes", (1, 0)),
                auto_filter=format_kwargs.get("auto_filter", True),
            )
        elif output_format == "json":
            return data.to_json(orient="records", indent=2).encode("utf-8")
        elif output_format == "jsonl":
            return data.to_json(orient="records", lines=True).encode("utf-8")
        else:
            raise ValueError(f"Unsupported export format: {output_format}")


# Module-level serialization helpers (run inside executor)

def _serialize_csv(
    data: pd.DataFrame,
    delimiter: str,
    encoding: str,
    include_index: bool,
    date_format: str | None,
) -> bytes:
    """Serialize DataFrame to CSV bytes.

    Args:
        data: DataFrame to serialize.
        delimiter: Field separator.
        encoding: Character encoding.
        include_index: Whether to include row index.
        date_format: strftime format for datetime columns.

    Returns:
        CSV bytes.
    """
    csv_str = data.to_csv(
        index=include_index,
        sep=delimiter,
        encoding=encoding,
        date_format=date_format,
    )
    return csv_str.encode(encoding)


def _serialize_parquet(
    data: pd.DataFrame,
    compression: str | None,
    row_group_size: int | None,
) -> bytes:
    """Serialize DataFrame to Parquet bytes.

    Args:
        data: DataFrame to serialize.
        compression: Parquet compression codec.
        row_group_size: Optional row group size.

    Returns:
        Parquet bytes.
    """
    buffer = io.BytesIO()
    kwargs: dict[str, Any] = {"engine": "pyarrow", "index": False}
    if compression:
        kwargs["compression"] = compression
    if row_group_size is not None:
        kwargs["row_group_size"] = row_group_size
    data.to_parquet(buffer, **kwargs)
    return buffer.getvalue()


def _serialize_excel(
    data: pd.DataFrame,
    sheet_name: str,
    freeze_panes: tuple[int, int],
    auto_filter: bool,
) -> bytes:
    """Serialize DataFrame to Excel .xlsx bytes.

    Args:
        data: DataFrame to serialize.
        sheet_name: Worksheet name.
        freeze_panes: Row/column coordinates to freeze.
        auto_filter: Whether to enable auto-filtering.

    Returns:
        Excel .xlsx bytes.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]

        if freeze_panes:
            row, col = freeze_panes
            worksheet.freeze_panes = worksheet.cell(row=row + 1, column=col + 1)

        if auto_filter:
            worksheet.auto_filter.ref = worksheet.dimensions

    return buffer.getvalue()


def _dict_to_json_bytes(data: dict[str, Any]) -> bytes:
    """Serialize a dict to indented JSON bytes.

    Args:
        data: Dict to serialize.

    Returns:
        JSON-encoded bytes.
    """
    import json

    return json.dumps(data, indent=2, default=str).encode("utf-8")


def _default_filename(output_format: ExportFormat) -> str:
    """Return the default output filename for a given format.

    Args:
        output_format: Export format string.

    Returns:
        Default filename string.
    """
    extension_map = {
        "csv": "output.csv",
        "parquet": "output.parquet",
        "excel": "output.xlsx",
        "json": "output.json",
        "jsonl": "output.jsonl",
    }
    return extension_map.get(output_format, f"output.{output_format}")


def _format_extension(output_format: ExportFormat) -> str:
    """Return the file extension for a given format.

    Args:
        output_format: Export format string.

    Returns:
        File extension including the leading dot.
    """
    extension_map = {
        "csv": ".csv",
        "parquet": ".parquet",
        "excel": ".xlsx",
        "json": ".json",
        "jsonl": ".jsonl",
    }
    return extension_map.get(output_format, f".{output_format}")

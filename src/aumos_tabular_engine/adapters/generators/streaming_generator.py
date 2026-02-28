"""Streaming CTGAN generator for large datasets that exceed available RAM.

Processes training data in configurable chunks using PyArrow Parquet
batch reading, enabling synthesis at arbitrary dataset sizes without OOM.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from typing import Any

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class StreamingCTGANGenerator:
    """Chunked CTGAN training and generation for large datasets.

    Processes input in chunks of `chunk_size` rows to control memory usage.
    Uses CTGAN's internal mini-batch mechanism with each chunk as a training
    epoch group. Output is generated in configurable chunk sizes and returned
    as a list of part URIs for multi-part Parquet assembly.

    Args:
        chunk_size: Rows per training chunk (default: 100_000).
        output_chunk_size: Rows per output Parquet file part (default: 500_000).
        epochs: CTGAN training epochs per chunk.
        batch_size: CTGAN batch size for training.
    """

    def __init__(
        self,
        chunk_size: int = 100_000,
        output_chunk_size: int = 500_000,
        epochs: int = 300,
        batch_size: int = 500,
    ) -> None:
        self._chunk_size = chunk_size
        self._output_chunk_size = output_chunk_size
        self._epochs = epochs
        self._batch_size = batch_size
        self._synthesizer: Any = None

    async def train_streaming(
        self,
        source_path: str,
    ) -> None:
        """Train CTGAN by streaming input Parquet from a local or staged path.

        Args:
            source_path: Local filesystem path to the Parquet file (pre-staged from MinIO).

        Raises:
            ImportError: If pyarrow or ctgan are not installed.
            FileNotFoundError: If source_path does not exist.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("pyarrow is required for streaming generation") from exc

        pq_file = pq.ParquetFile(source_path)
        chunk_count = 0

        for batch in pq_file.iter_batches(batch_size=self._chunk_size):
            chunk_df = batch.to_pandas()
            await asyncio.to_thread(self._train_chunk, chunk_df)
            chunk_count += 1
            logger.info(
                "streaming_chunk_trained",
                chunk_index=chunk_count,
                chunk_rows=len(chunk_df),
            )

        logger.info(
            "streaming_training_complete",
            total_chunks=chunk_count,
        )

    def _train_chunk(self, chunk: pd.DataFrame) -> None:
        """Incremental CTGAN training on a single chunk.

        Args:
            chunk: DataFrame chunk to train on.
        """
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
        except ImportError as exc:
            raise ImportError("sdv is required for CTGAN streaming generation") from exc

        if self._synthesizer is None:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(chunk)
            self._synthesizer = CTGANSynthesizer(
                metadata=metadata,
                epochs=self._epochs,
                batch_size=self._batch_size,
            )
            self._synthesizer.fit(chunk)
        else:
            self._synthesizer.fit(chunk)

    async def generate_streaming(
        self,
        total_rows: int,
        job_id: uuid.UUID,
    ) -> tuple[list[pd.DataFrame], list[str]]:
        """Generate total_rows in chunks and return as a list of DataFrames with part keys.

        Args:
            total_rows: Total synthetic rows to generate.
            job_id: Generation job UUID for output key naming.

        Returns:
            Tuple of (list of DataFrame chunks, list of part key strings).

        Raises:
            RuntimeError: If model has not been trained.
        """
        if self._synthesizer is None:
            raise RuntimeError(
                "StreamingCTGANGenerator must be trained before generating. "
                "Call train_streaming() first."
            )

        parts: list[pd.DataFrame] = []
        part_keys: list[str] = []
        generated_rows = 0
        part_index = 0

        while generated_rows < total_rows:
            rows_in_chunk = min(self._output_chunk_size, total_rows - generated_rows)
            chunk: pd.DataFrame = await asyncio.to_thread(
                self._synthesizer.sample, num_rows=rows_in_chunk
            )
            key = f"tab-jobs/{job_id}/output_part_{part_index:04d}.parquet"
            parts.append(chunk)
            part_keys.append(key)
            generated_rows += rows_in_chunk
            part_index += 1

            logger.info(
                "streaming_chunk_generated",
                part_index=part_index,
                rows_in_chunk=rows_in_chunk,
                total_generated=generated_rows,
                total_target=total_rows,
            )

        return parts, part_keys

    def is_trained(self) -> bool:
        """Return True if the model has been trained.

        Returns:
            True if train_streaming() has been called successfully.
        """
        return self._synthesizer is not None

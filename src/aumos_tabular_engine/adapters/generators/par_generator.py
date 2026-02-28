"""PAR (Probabilistic AutoRegressive) time-series generator adapter.

Wraps SDV PARSynthesizer to generate synthetic time-series tabular data
that preserves temporal ordering and autocorrelation structure.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class PARGenerator:
    """Probabilistic AutoRegressive synthesizer adapter for temporal tabular data.

    Wraps SDV PARSynthesizer to generate time-series rows preserving temporal
    ordering and autocorrelation structure. Suitable for financial transaction
    histories, EHR visit sequences, and IoT sensor readings.

    Args:
        epochs: Number of training epochs for the PAR model.
        sample_size: Number of sequences to generate per call.
        context_columns: List of columns defining sequence groups (e.g., entity_id).
        verbose: Whether to print training progress.
    """

    def __init__(
        self,
        epochs: int = 128,
        sample_size: int = 100,
        context_columns: list[str] | None = None,
        verbose: bool = False,
    ) -> None:
        self._epochs = epochs
        self._sample_size = sample_size
        self._context_columns: list[str] = context_columns or []
        self._verbose = verbose
        self._synthesizer: Any = None

    async def train(
        self,
        real_data: pd.DataFrame,
        sequence_key: str,
        sequence_index: str,
    ) -> None:
        """Train the PAR model on real sequential data.

        Args:
            real_data: DataFrame with sequence_key and sequence_index columns.
            sequence_key: Column identifying each sequence (e.g., customer_id).
            sequence_index: Column defining temporal order (e.g., transaction_date).

        Raises:
            ImportError: If sdv[sequential] is not installed.
            ValueError: If sequence_key or sequence_index columns are missing.
        """
        if sequence_key not in real_data.columns:
            raise ValueError(f"sequence_key '{sequence_key}' not found in data columns")
        if sequence_index not in real_data.columns:
            raise ValueError(f"sequence_index '{sequence_index}' not found in data columns")

        try:
            from sdv.sequential import PARSynthesizer
            from sdv.metadata import SequentialMetadata
        except ImportError as exc:
            raise ImportError(
                "sdv[sequential] is required for PAR generation. "
                "Install with: pip install 'sdv[sequential]>=1.10'"
            ) from exc

        metadata = SequentialMetadata()
        metadata.detect_from_dataframe(
            data=real_data,
            sequence_key=sequence_key,
            sequence_index=sequence_index,
        )

        synthesizer = PARSynthesizer(
            metadata=metadata,
            epochs=self._epochs,
            context_columns=self._context_columns,
            verbose=self._verbose,
        )

        await asyncio.to_thread(synthesizer.fit, real_data)
        self._synthesizer = synthesizer

        logger.info(
            "par_generator_trained",
            epochs=self._epochs,
            rows=len(real_data),
            sequence_key=sequence_key,
            sequence_index=sequence_index,
        )

    async def generate(self, num_sequences: int) -> pd.DataFrame:
        """Generate synthetic sequential data.

        Args:
            num_sequences: Number of independent sequences to generate.

        Returns:
            DataFrame with synthetic time-series rows preserving temporal ordering.

        Raises:
            RuntimeError: If model has not been trained via train().
        """
        if self._synthesizer is None:
            raise RuntimeError(
                "PARGenerator must be trained before generating. Call train() first."
            )

        result: pd.DataFrame = await asyncio.to_thread(
            self._synthesizer.sample, num_sequences=num_sequences
        )

        logger.info(
            "par_generator_generated",
            num_sequences=num_sequences,
            rows_generated=len(result),
        )
        return result

    def is_trained(self) -> bool:
        """Return True if the model has been trained.

        Returns:
            True if train() has been called successfully, False otherwise.
        """
        return self._synthesizer is not None

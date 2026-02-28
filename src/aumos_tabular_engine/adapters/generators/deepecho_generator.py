"""DeepEcho LSTM-based time-series generator adapter.

Wraps the DeepEcho PARModel for long-dependency sequence synthesis
where PAR's autoregressive window is insufficient.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class DeepEchoGenerator:
    """DeepEcho LSTM-based time-series generator for long-dependency sequences.

    Uses the deepecho PARModel to train on sequences and generate new synthetic
    sequences. Suitable for time-series with 500+ steps where PAR's limited
    autoregressive window fails to capture long-range dependencies.

    Args:
        epochs: LSTM training epochs.
        max_sequence_len: Maximum sequence length to model.
        segment_size: Segment size for LSTM state management.
    """

    def __init__(
        self,
        epochs: int = 256,
        max_sequence_len: int = 500,
        segment_size: int = 50,
    ) -> None:
        self._epochs = epochs
        self._max_sequence_len = max_sequence_len
        self._segment_size = segment_size
        self._model: Any = None

    async def train(
        self,
        sequences: list[pd.DataFrame],
        data_types: list[tuple[str, str]] | None = None,
    ) -> None:
        """Train on a list of sequence DataFrames.

        Args:
            sequences: List of DataFrames, each representing one sequence.
                All DataFrames must have identical columns.
            data_types: Optional column type descriptors as list of (col_name, type_str).
                If None, types are inferred automatically.

        Raises:
            ImportError: If deepecho is not installed.
            ValueError: If sequences list is empty.
        """
        if not sequences:
            raise ValueError("sequences must be a non-empty list of DataFrames")

        try:
            from deepecho import PARModel
        except ImportError as exc:
            raise ImportError(
                "deepecho is required for DeepEcho generation. "
                "Install with: pip install deepecho>=0.4"
            ) from exc

        model = PARModel(
            epochs=self._epochs,
            max_sequence_len=self._max_sequence_len,
            segment_size=self._segment_size,
        )

        await asyncio.to_thread(model.fit, sequences, data_types=data_types)
        self._model = model

        logger.info(
            "deepecho_generator_trained",
            epochs=self._epochs,
            num_sequences=len(sequences),
            max_sequence_len=self._max_sequence_len,
        )

    async def generate(self, num_sequences: int) -> list[pd.DataFrame]:
        """Generate synthetic sequences.

        Args:
            num_sequences: Number of independent sequences to generate.

        Returns:
            List of DataFrames, each representing one synthetic sequence.

        Raises:
            RuntimeError: If model has not been trained via train().
        """
        if self._model is None:
            raise RuntimeError(
                "DeepEchoGenerator must be trained before generating. Call train() first."
            )

        result: list[pd.DataFrame] = await asyncio.to_thread(
            self._model.sample, num_sequences=num_sequences
        )

        logger.info(
            "deepecho_generator_generated",
            num_sequences=num_sequences,
        )
        return result

    def is_trained(self) -> bool:
        """Return True if the model has been trained.

        Returns:
            True if train() has been called successfully, False otherwise.
        """
        return self._model is not None

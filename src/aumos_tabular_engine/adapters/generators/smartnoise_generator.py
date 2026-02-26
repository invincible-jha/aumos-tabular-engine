"""SmartNoise differential privacy generator for aumos-tabular-engine.

Wraps SmartNoise-Synth behind the GeneratorProtocol to provide formal
(epsilon, delta)-differential privacy guarantees. All DP jobs must call
the privacy engine to allocate and then consume epsilon budget.
"""

import asyncio
from functools import partial
from typing import Any

import pandas as pd

from aumos_common.errors import AumOSError, ErrorCode
from aumos_common.observability import get_logger

logger = get_logger(__name__)

_STATUS_UNTRAINED = "untrained"
_STATUS_TRAINING = "training"
_STATUS_READY = "ready"
_STATUS_ERROR = "error"

# SmartNoise synthesizer name constants
_SYNTHESIZER_MWEM = "mwem"
_SYNTHESIZER_DPCTGAN = "dpctgan"
_SYNTHESIZER_PATECTGAN = "patectgan"
_SYNTHESIZER_MST = "mst"
_SYNTHESIZER_AIM = "aim"

_DEFAULT_SYNTHESIZER = _SYNTHESIZER_DPCTGAN


class SmartNoiseGenerator:
    """SmartNoise-Synth DP generator implementing GeneratorProtocol.

    Provides formal (epsilon, delta)-differential privacy guarantees using
    the SmartNoise Synthesizer library. Supports multiple DP mechanisms:

    - dpctgan: CTGAN with DP-SGD gradient clipping (good general-purpose DP synthesizer)
    - patectgan: PATE-based CTGAN (teacher-ensemble DP; better for larger datasets)
    - mwem: Multiplicative Weights Exponential Mechanism (fast, good for low-dimensional)
    - mst: Maximum Spanning Tree (marginals-based, excellent privacy-utility tradeoff)
    - aim: Adaptive and Iterative Mechanism (state-of-the-art DP synth)

    IMPORTANT: The caller is responsible for:
    1. Calling aumos-privacy-engine to allocate budget BEFORE training
    2. Calling aumos-privacy-engine to consume budget AFTER successful generation
    3. Passing the allocation_id for audit trail purposes
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        synthesizer_name: str = _DEFAULT_SYNTHESIZER,
        batch_size: int = 500,
        epochs: int = 300,
        sigma: float = 1.0,
        max_per_sample_grad_norm: float = 1.0,
        sample_rate: float = 0.1,
        cuda: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialise the SmartNoise DP generator.

        Args:
            epsilon: Privacy budget epsilon (lower = more private, higher = more utility).
                     Typical values: 0.1 (strong privacy) to 10.0 (weak privacy).
            delta: Privacy relaxation delta. Should be << 1/n where n is dataset size.
                   Typical value: 1e-5.
            synthesizer_name: SmartNoise synthesizer to use (dpctgan/patectgan/mwem/mst/aim).
            batch_size: Training batch size (for CTGAN-based synthesizers).
            epochs: Number of training epochs.
            sigma: Gaussian noise multiplier for DP-SGD (dpctgan).
            max_per_sample_grad_norm: Per-sample gradient clipping bound.
            sample_rate: Fraction of dataset used per training batch (Poisson sampling).
            cuda: Whether to use GPU acceleration.
            verbose: Whether to print training progress.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        self._epsilon = epsilon
        self._delta = delta
        self._synthesizer_name = synthesizer_name
        self._batch_size = batch_size
        self._epochs = epochs
        self._sigma = sigma
        self._max_per_sample_grad_norm = max_per_sample_grad_norm
        self._sample_rate = sample_rate
        self._cuda = cuda
        self._verbose = verbose
        self._synthesizer: Any | None = None
        self._status = _STATUS_UNTRAINED

    async def train(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        epochs: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Train the SmartNoise DP synthesizer on real data.

        CRITICAL: The privacy engine budget allocation MUST be performed by the
        caller (GenerationService) BEFORE invoking this method. This method does
        NOT call the privacy engine directly — that is handled in the service layer
        to ensure proper budget accounting even on training failures.

        Args:
            data: Real source DataFrame to train on.
            metadata: SDV-compatible SingleTableMetadata dict.
            epochs: Override default epoch count.
            batch_size: Override default batch size.
            **kwargs: Additional synthesizer-specific kwargs.

        Raises:
            AumOSError: If the DP synthesizer fails to train.
        """
        from snsynth import Synthesizer

        self._status = _STATUS_TRAINING
        training_epochs = epochs or self._epochs
        training_batch_size = batch_size or self._batch_size

        # Identify categorical columns from metadata for SmartNoise
        categorical_columns = _extract_categorical_columns(metadata)

        logger.info(
            "SmartNoise DP training started",
            synthesizer=self._synthesizer_name,
            epsilon=self._epsilon,
            delta=self._delta,
            num_rows=len(data),
            num_cols=len(data.columns),
            categorical_count=len(categorical_columns),
            epochs=training_epochs,
        )

        synthesizer_kwargs: dict[str, Any] = {
            "epsilon": self._epsilon,
            "verbose": self._verbose,
        }

        if self._synthesizer_name in (_SYNTHESIZER_DPCTGAN, _SYNTHESIZER_PATECTGAN):
            synthesizer_kwargs.update(
                {
                    "batch_size": training_batch_size,
                    "epochs": training_epochs,
                    "sigma": self._sigma,
                    "max_per_sample_grad_norm": self._max_per_sample_grad_norm,
                    "sample_rate": self._sample_rate,
                    "cuda": self._cuda,
                    "delta": self._delta,
                }
            )
        elif self._synthesizer_name in (_SYNTHESIZER_MST, _SYNTHESIZER_AIM):
            synthesizer_kwargs["delta"] = self._delta

        synthesizer = Synthesizer.create(self._synthesizer_name, **synthesizer_kwargs)

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                partial(
                    synthesizer.fit,
                    data,
                    categorical_columns=categorical_columns,
                ),
            )
        except Exception as exc:
            self._status = _STATUS_ERROR
            logger.error(
                "SmartNoise DP training failed",
                synthesizer=self._synthesizer_name,
                epsilon=self._epsilon,
                error=str(exc),
            )
            raise AumOSError(
                message=f"SmartNoise training failed: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
            ) from exc

        self._synthesizer = synthesizer
        self._status = _STATUS_READY
        logger.info(
            "SmartNoise DP training complete",
            synthesizer=self._synthesizer_name,
            epsilon=self._epsilon,
            delta=self._delta,
        )

    async def generate(
        self,
        num_rows: int,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate differentially private synthetic rows.

        Args:
            num_rows: Number of synthetic rows to generate.
            conditions: Optional conditional constraints (not all DP synthesizers support this).

        Returns:
            DataFrame with `num_rows` DP-protected synthetic rows.

        Raises:
            RuntimeError: If called before training.
            AumOSError: If generation fails.
        """
        if self._synthesizer is None:
            raise RuntimeError("SmartNoiseGenerator must be trained before generating data.")

        logger.info(
            "SmartNoise DP generating",
            num_rows=num_rows,
            epsilon=self._epsilon,
        )
        loop = asyncio.get_event_loop()

        try:
            synthetic = await loop.run_in_executor(
                None,
                partial(self._synthesizer.sample, num_rows),
            )
        except Exception as exc:
            logger.error("SmartNoise generation failed", error=str(exc))
            raise AumOSError(
                message=f"SmartNoise generation failed: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
            ) from exc

        logger.info(
            "SmartNoise DP generation complete",
            num_rows=len(synthetic),
            epsilon_consumed=self._epsilon,
        )
        return synthetic

    def get_status(self) -> str:
        """Return current generator status.

        Returns:
            One of: 'untrained', 'training', 'ready', 'error'.
        """
        return self._status

    @property
    def epsilon(self) -> float:
        """Return the configured epsilon for this generator.

        Returns:
            Configured epsilon value.
        """
        return self._epsilon

    @property
    def delta(self) -> float:
        """Return the configured delta for this generator.

        Returns:
            Configured delta value.
        """
        return self._delta


def _extract_categorical_columns(metadata: dict[str, Any]) -> list[str]:
    """Extract categorical column names from SDV metadata.

    Args:
        metadata: SDV SingleTableMetadata dict with 'columns' key.

    Returns:
        List of column names with categorical/boolean sdtype.
    """
    categorical_sdtypes = {"categorical", "boolean", "id"}
    columns: dict[str, Any] = metadata.get("columns", {})
    return [
        col_name
        for col_name, col_meta in columns.items()
        if col_meta.get("sdtype") in categorical_sdtypes
    ]

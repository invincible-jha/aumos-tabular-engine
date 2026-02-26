"""SDV-based generator adapters for aumos-tabular-engine.

Wraps SDV's GaussianCopulaSynthesizer and TVAESynthesizer behind the
GeneratorProtocol to enable dependency inversion and easy backend swapping.
"""

import asyncio
from functools import partial
from typing import Any

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Generator status constants
_STATUS_UNTRAINED = "untrained"
_STATUS_TRAINING = "training"
_STATUS_READY = "ready"
_STATUS_ERROR = "error"


class GaussianCopulaGenerator:
    """SDV GaussianCopulaSynthesizer adapter implementing GeneratorProtocol.

    GaussianCopula is a fast statistical synthesizer that models column-to-column
    correlations via copula distributions. Suitable for smaller datasets or
    when training speed is more important than deep feature learning.
    """

    def __init__(
        self,
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        numerical_distributions: dict[str, str] | None = None,
        default_distribution: str = "beta",
    ) -> None:
        """Initialise the GaussianCopula generator.

        Args:
            enforce_min_max_values: Clip generated values to observed min/max.
            enforce_rounding: Round numerical columns to observed precision.
            numerical_distributions: Per-column distribution overrides.
            default_distribution: Default distribution for numerical columns.
        """
        self._enforce_min_max_values = enforce_min_max_values
        self._enforce_rounding = enforce_rounding
        self._numerical_distributions = numerical_distributions or {}
        self._default_distribution = default_distribution
        self._synthesizer: Any | None = None
        self._status = _STATUS_UNTRAINED

    async def train(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Train the GaussianCopula synthesizer on real data.

        Runs SDV training in a thread pool executor to avoid blocking
        the async event loop during CPU-bound fitting.

        Args:
            data: Real source DataFrame.
            metadata: SDV SingleTableMetadata dict.
            **kwargs: Ignored (no training hyperparameters for GaussianCopula).
        """
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer

        self._status = _STATUS_TRAINING
        logger.info(
            "GaussianCopula training started",
            num_rows=len(data),
            num_cols=len(data.columns),
        )

        sdv_metadata = SingleTableMetadata.load_from_dict(metadata)

        synthesizer = GaussianCopulaSynthesizer(
            metadata=sdv_metadata,
            enforce_min_max_values=self._enforce_min_max_values,
            enforce_rounding=self._enforce_rounding,
            numerical_distributions=self._numerical_distributions,
            default_distribution=self._default_distribution,
        )

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, partial(synthesizer.fit, data))
        except Exception as exc:
            self._status = _STATUS_ERROR
            logger.error("GaussianCopula training failed", error=str(exc))
            raise

        self._synthesizer = synthesizer
        self._status = _STATUS_READY
        logger.info("GaussianCopula training complete")

    async def generate(
        self,
        num_rows: int,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic rows using the trained GaussianCopula model.

        Args:
            num_rows: Number of rows to generate.
            conditions: Optional conditional generation dict (column → value).

        Returns:
            DataFrame with `num_rows` synthetic rows.

        Raises:
            RuntimeError: If called before training.
        """
        if self._synthesizer is None:
            raise RuntimeError("GaussianCopulaGenerator must be trained before generating data.")

        logger.info("GaussianCopula generating", num_rows=num_rows)
        loop = asyncio.get_event_loop()

        if conditions:
            from sdv.sampling import Condition

            condition = Condition(num_rows=num_rows, column_values=conditions)
            synthetic = await loop.run_in_executor(
                None,
                partial(self._synthesizer.sample_from_conditions, [condition]),
            )
        else:
            synthetic = await loop.run_in_executor(
                None,
                partial(self._synthesizer.sample, num_rows=num_rows),
            )

        logger.info("GaussianCopula generation complete", num_rows=len(synthetic))
        return synthetic

    def get_status(self) -> str:
        """Return current generator status.

        Returns:
            One of: 'untrained', 'training', 'ready', 'error'.
        """
        return self._status


class TVAEGenerator:
    """SDV TVAESynthesizer adapter implementing GeneratorProtocol.

    TVAE (Tabular Variational Autoencoder) is a deep learning synthesizer that
    learns a latent representation of the data via a VAE architecture.
    Typically achieves better fidelity than GaussianCopula for complex datasets.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        compress_dims: tuple[int, ...] = (128, 128),
        decompress_dims: tuple[int, ...] = (128, 128),
        l2scale: float = 1e-5,
        batch_size: int = 500,
        epochs: int = 300,
        loss_factor: int = 2,
        cuda: bool = False,
    ) -> None:
        """Initialise the TVAE generator.

        Args:
            embedding_dim: Dimensionality of the VAE latent space.
            compress_dims: Encoder hidden layer dimensions.
            decompress_dims: Decoder hidden layer dimensions.
            l2scale: L2 regularisation weight.
            batch_size: Training batch size.
            epochs: Number of training epochs.
            loss_factor: Reconstruction loss weighting.
            cuda: Whether to use GPU acceleration.
        """
        self._embedding_dim = embedding_dim
        self._compress_dims = compress_dims
        self._decompress_dims = decompress_dims
        self._l2scale = l2scale
        self._batch_size = batch_size
        self._epochs = epochs
        self._loss_factor = loss_factor
        self._cuda = cuda
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
        """Train the TVAE synthesizer on real data.

        Args:
            data: Real source DataFrame.
            metadata: SDV SingleTableMetadata dict.
            epochs: Override default epoch count.
            batch_size: Override default batch size.
            **kwargs: Additional SDV training kwargs.
        """
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import TVAESynthesizer

        self._status = _STATUS_TRAINING
        training_epochs = epochs or self._epochs
        training_batch_size = batch_size or self._batch_size

        logger.info(
            "TVAE training started",
            num_rows=len(data),
            num_cols=len(data.columns),
            epochs=training_epochs,
            batch_size=training_batch_size,
        )

        sdv_metadata = SingleTableMetadata.load_from_dict(metadata)
        synthesizer = TVAESynthesizer(
            metadata=sdv_metadata,
            embedding_dim=self._embedding_dim,
            compress_dims=self._compress_dims,
            decompress_dims=self._decompress_dims,
            l2scale=self._l2scale,
            batch_size=training_batch_size,
            epochs=training_epochs,
            loss_factor=self._loss_factor,
            cuda=self._cuda,
        )

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, partial(synthesizer.fit, data))
        except Exception as exc:
            self._status = _STATUS_ERROR
            logger.error("TVAE training failed", error=str(exc))
            raise

        self._synthesizer = synthesizer
        self._status = _STATUS_READY
        logger.info("TVAE training complete", epochs=training_epochs)

    async def generate(
        self,
        num_rows: int,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic rows using the trained TVAE model.

        Args:
            num_rows: Number of rows to generate.
            conditions: Optional conditional generation dict.

        Returns:
            DataFrame with `num_rows` synthetic rows.

        Raises:
            RuntimeError: If called before training.
        """
        if self._synthesizer is None:
            raise RuntimeError("TVAEGenerator must be trained before generating data.")

        logger.info("TVAE generating", num_rows=num_rows)
        loop = asyncio.get_event_loop()
        synthetic = await loop.run_in_executor(
            None,
            partial(self._synthesizer.sample, num_rows=num_rows),
        )
        logger.info("TVAE generation complete", num_rows=len(synthetic))
        return synthetic

    def get_status(self) -> str:
        """Return current generator status.

        Returns:
            One of: 'untrained', 'training', 'ready', 'error'.
        """
        return self._status

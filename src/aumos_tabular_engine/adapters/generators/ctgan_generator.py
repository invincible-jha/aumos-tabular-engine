"""CTGAN generator adapter for aumos-tabular-engine.

Wraps the CTGAN (Conditional Tabular GAN) deep learning synthesizer behind the
GeneratorProtocol. CTGAN uses a conditional GAN architecture specifically designed
for tabular data, handling mixed-type columns and imbalanced categorical distributions.
"""

import asyncio
from functools import partial
from typing import Any

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_STATUS_UNTRAINED = "untrained"
_STATUS_TRAINING = "training"
_STATUS_READY = "ready"
_STATUS_ERROR = "error"


class CTGANGenerator:
    """CTGAN deep learning generator implementing GeneratorProtocol.

    CTGAN (Conditional Tabular GAN) is a GAN-based synthesizer designed for
    tabular data. It uses mode-specific normalisation for continuous columns and
    a conditional vector to handle imbalanced categorical distributions.

    Best suited for datasets with:
    - Mixed numeric and categorical columns
    - Imbalanced categorical distributions
    - Complex non-linear relationships between columns
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        generator_dim: tuple[int, ...] = (256, 256),
        discriminator_dim: tuple[int, ...] = (256, 256),
        generator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-6,
        batch_size: int = 500,
        discriminator_steps: int = 1,
        log_frequency: bool = True,
        verbose: bool = False,
        epochs: int = 300,
        pac: int = 10,
        cuda: bool = False,
    ) -> None:
        """Initialise CTGAN with configurable GAN architecture hyperparameters.

        Args:
            embedding_dim: Dimensionality of the embedding for the random noise input.
            generator_dim: Hidden layer sizes for the generator network.
            discriminator_dim: Hidden layer sizes for the discriminator network.
            generator_lr: Generator Adam learning rate.
            generator_decay: Generator Adam weight decay.
            discriminator_lr: Discriminator Adam learning rate.
            discriminator_decay: Discriminator Adam weight decay.
            batch_size: Training mini-batch size.
            discriminator_steps: Number of discriminator updates per generator step.
            log_frequency: Whether to log loss at each epoch.
            verbose: Whether to print training progress.
            epochs: Number of training epochs.
            pac: Number of samples per PAC (Packed Conditional Network) group.
            cuda: Whether to use GPU acceleration via CUDA.
        """
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay
        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self._pac = pac
        self._cuda = cuda
        self._synthesizer: Any | None = None
        self._status = _STATUS_UNTRAINED
        self._trained_columns: list[str] = []

    async def train(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        epochs: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Train CTGAN on real tabular data.

        Runs SDV's CTGANSynthesizer.fit() in a thread pool executor to avoid
        blocking the async event loop during GPU/CPU-intensive training.

        Args:
            data: Real source DataFrame to learn from.
            metadata: SDV-compatible SingleTableMetadata dict.
            epochs: Override default epoch count for this run.
            batch_size: Override default batch size for this run.
            **kwargs: Additional SDV training kwargs.
        """
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer

        self._status = _STATUS_TRAINING
        training_epochs = epochs or self._epochs
        training_batch_size = batch_size or self._batch_size

        logger.info(
            "CTGAN training started",
            num_rows=len(data),
            num_cols=len(data.columns),
            epochs=training_epochs,
            batch_size=training_batch_size,
            cuda_enabled=self._cuda,
        )

        sdv_metadata = SingleTableMetadata.load_from_dict(metadata)
        synthesizer = CTGANSynthesizer(
            metadata=sdv_metadata,
            embedding_dim=self._embedding_dim,
            generator_dim=self._generator_dim,
            discriminator_dim=self._discriminator_dim,
            generator_lr=self._generator_lr,
            generator_decay=self._generator_decay,
            discriminator_lr=self._discriminator_lr,
            discriminator_decay=self._discriminator_decay,
            batch_size=training_batch_size,
            discriminator_steps=self._discriminator_steps,
            log_frequency=self._log_frequency,
            verbose=self._verbose,
            epochs=training_epochs,
            pac=self._pac,
            cuda=self._cuda,
        )

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, partial(synthesizer.fit, data))
        except Exception as exc:
            self._status = _STATUS_ERROR
            logger.error("CTGAN training failed", error=str(exc))
            raise

        self._synthesizer = synthesizer
        self._trained_columns = list(data.columns)
        self._status = _STATUS_READY
        logger.info(
            "CTGAN training complete",
            epochs=training_epochs,
            num_columns=len(self._trained_columns),
        )

    async def generate(
        self,
        num_rows: int,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic tabular data using the trained CTGAN model.

        For conditional generation, uses SDV's Condition object to constrain
        the generated data to specified column values.

        Args:
            num_rows: Number of synthetic rows to produce.
            conditions: Optional dict of column → fixed value for conditional generation.

        Returns:
            DataFrame with `num_rows` rows matching the trained schema.

        Raises:
            RuntimeError: If called before train().
        """
        if self._synthesizer is None:
            raise RuntimeError("CTGANGenerator must be trained before generating data.")

        logger.info("CTGAN generating", num_rows=num_rows, conditional=conditions is not None)
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

        logger.info("CTGAN generation complete", num_rows=len(synthetic))
        return synthetic

    def get_status(self) -> str:
        """Return current generator status.

        Returns:
            One of: 'untrained', 'training', 'ready', 'error'.
        """
        return self._status

    def get_loss_values(self) -> list[dict[str, float]]:
        """Return recorded training loss values if available.

        Returns:
            List of dicts with 'epoch', 'generator_loss', 'discriminator_loss' keys.
            Empty list if not yet trained or loss logging disabled.
        """
        if self._synthesizer is None:
            return []

        try:
            loss_values = self._synthesizer.get_loss_values()
            return loss_values.to_dict("records") if hasattr(loss_values, "to_dict") else []
        except Exception:
            return []

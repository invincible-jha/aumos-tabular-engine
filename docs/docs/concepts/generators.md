# Generators

The tabular engine supports 7 generator backends. Choose based on your data type and privacy requirements.

## CTGAN

Conditional Tabular GAN — a deep learning generator that conditions on discrete columns to handle
mixed data types. Best for datasets with complex correlations and non-Gaussian distributions.

**When to use:** Financial transactions, customer records with mixed categorical/numerical columns.

**Key parameters:**
- `epochs` (default: 300) — more epochs = better quality, longer training
- `batch_size` (default: 500) — reduce for low-memory environments

## TVAE

Tabular Variational Autoencoder — encodes tabular data into a latent space and decodes to
generate new rows. Faster than CTGAN with competitive quality.

**When to use:** Structured tables with clear column semantics, when speed matters.

## Gaussian Copula

Statistical generator that captures pairwise correlations via copulas. No neural network.
Fastest generator; best for continuous numerical data.

**When to use:** Rapid prototyping, simple numerical datasets, when training time is critical.

## SmartNoise

Differentially private generator backed by OpenDP. Every generation job must provide an
epsilon budget; the engine deducts from the tenant's total budget via `aumos-privacy-engine`.

**When to use:** Any regulated data (healthcare, finance) where formal privacy guarantees are required.

See [Privacy Budget](privacy-budget.md) for epsilon allocation details.

## HMA (Multi-Table)

Hierarchical Multi-table Aggregate — generates multiple tables simultaneously while preserving
foreign key cardinalities and referential integrity.

**When to use:** Multi-table schemas (e.g., customers + orders + order_items).

## PAR (Time-Series)

Probabilistic AutoRegressive synthesizer from SDV. Trained on sequential data with a
`sequence_key` (entity ID) and `sequence_index` (timestamp column).

**When to use:** Financial transaction histories, EHR visit sequences, IoT sensor readings.

## DeepEcho (Time-Series)

LSTM-based time-series generator for longer dependency sequences (500+ steps).

**When to use:** Long time-series where PAR's autoregressive window is insufficient.

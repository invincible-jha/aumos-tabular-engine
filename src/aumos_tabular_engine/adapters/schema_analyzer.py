"""Schema analyzer adapter for auto-detecting column types and distributions.

Uses pandas to profile real datasets: detect column types, compute value
distributions, identify nullability, infer constraints, and detect
potential foreign key candidate columns.
"""

from typing import Any

import numpy as np
import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# SDV sdtype mapping constants
_SDTYPE_NUMERICAL = "numerical"
_SDTYPE_CATEGORICAL = "categorical"
_SDTYPE_BOOLEAN = "boolean"
_SDTYPE_DATETIME = "datetime"
_SDTYPE_ID = "id"
_SDTYPE_TEXT = "text"

# Threshold for treating a numeric column as categorical
_MAX_UNIQUE_RATIO_FOR_CATEGORICAL = 0.05
_MAX_UNIQUE_COUNT_FOR_CATEGORICAL = 50

# Threshold for detecting potential ID/PK columns
_MIN_UNIQUE_RATIO_FOR_ID = 0.99


class SchemaAnalyzer:
    """Auto-detect column types, distributions, and constraints from real data.

    Produces SDV-compatible SingleTableMetadata dicts augmented with
    distribution summaries and inferred constraint suggestions.
    """

    async def analyze(
        self,
        data: pd.DataFrame,
        sample_size: int | None = None,
    ) -> dict[str, Any]:
        """Analyze a DataFrame and return schema metadata.

        Profiles up to `sample_size` rows to detect column types, value
        distributions, nullability, potential ID columns, and suggests
        business rule constraints.

        Args:
            data: DataFrame to profile.
            sample_size: Optional maximum rows to sample.

        Returns:
            Schema metadata dict with:
            - columns: SDV-compatible column type specs
            - column_profiles: Distribution summaries per column
            - inferred_constraints: Auto-detected constraint suggestions
            - row_count: Total rows analyzed
            - null_columns: Columns with >50% null values
        """
        if sample_size is not None and len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)

        logger.info(
            "Analyzing schema",
            num_rows=len(data),
            num_columns=len(data.columns),
        )

        columns: dict[str, Any] = {}
        column_profiles: dict[str, Any] = {}
        inferred_constraints: list[dict[str, Any]] = []
        null_columns: list[str] = []

        for col_name in data.columns:
            series = data[col_name]
            null_fraction = series.isna().mean()
            unique_count = series.nunique()
            total_rows = len(series)
            unique_ratio = unique_count / total_rows if total_rows > 0 else 0.0

            # Track highly null columns
            if null_fraction > 0.5:
                null_columns.append(col_name)

            # Detect column sdtype
            sdtype, column_meta = _detect_sdtype(
                col_name=col_name,
                series=series,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
            )
            columns[col_name] = {"sdtype": sdtype, **column_meta}

            # Build distribution profile
            profile = _build_column_profile(
                series=series,
                sdtype=sdtype,
                null_fraction=null_fraction,
                unique_count=unique_count,
            )
            column_profiles[col_name] = profile

            # Infer constraints
            col_constraints = _infer_column_constraints(
                col_name=col_name,
                series=series,
                sdtype=sdtype,
                null_fraction=null_fraction,
            )
            inferred_constraints.extend(col_constraints)

        schema: dict[str, Any] = {
            "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1",
            "columns": columns,
            "column_profiles": column_profiles,
            "inferred_constraints": inferred_constraints,
            "row_count": len(data),
            "null_columns": null_columns,
        }

        logger.info(
            "Schema analysis complete",
            num_columns=len(columns),
            inferred_constraints=len(inferred_constraints),
            null_columns=len(null_columns),
        )
        return schema


def _detect_sdtype(
    col_name: str,
    series: pd.Series,
    unique_count: int,
    unique_ratio: float,
) -> tuple[str, dict[str, Any]]:
    """Determine the SDV sdtype and additional metadata for a column.

    Args:
        col_name: Column name for heuristic hints.
        series: Column data series.
        unique_count: Number of unique non-null values.
        unique_ratio: Ratio of unique values to total rows.

    Returns:
        Tuple of (sdtype, extra_metadata_dict).
    """
    dtype = series.dtype

    # Boolean columns
    if dtype == bool or (dtype == object and series.dropna().isin([True, False, 0, 1, "true", "false", "True", "False"]).all()):
        return _SDTYPE_BOOLEAN, {}

    # Datetime columns
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return _SDTYPE_DATETIME, {"datetime_format": "%Y-%m-%d %H:%M:%S"}

    # Try parsing string columns as datetime
    if dtype == object:
        try:
            pd.to_datetime(series.dropna().head(100), infer_datetime_format=True)
            return _SDTYPE_DATETIME, {"datetime_format": None}
        except (ValueError, TypeError):
            pass

    # Numeric columns
    if pd.api.types.is_numeric_dtype(dtype):
        # Check if it looks like an ID column (high cardinality integer)
        is_integer_like = pd.api.types.is_integer_dtype(dtype)
        col_lower = col_name.lower()
        is_id_named = any(
            kw in col_lower for kw in ["_id", "id_", "uuid", "key", "pk", "fk"]
        )

        if is_id_named and is_integer_like and unique_ratio >= _MIN_UNIQUE_RATIO_FOR_ID:
            return _SDTYPE_ID, {"regex_format": r"\d+"}

        # Low cardinality numeric → treat as categorical
        if unique_count <= _MAX_UNIQUE_COUNT_FOR_CATEGORICAL and unique_ratio < _MAX_UNIQUE_RATIO_FOR_CATEGORICAL:
            return _SDTYPE_CATEGORICAL, {}

        return _SDTYPE_NUMERICAL, {}

    # Object/string columns
    if dtype == object:
        col_lower = col_name.lower()

        # UUID-like column names
        if any(kw in col_lower for kw in ["uuid", "guid"]):
            return _SDTYPE_ID, {"regex_format": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"}

        # ID-named string columns
        if any(kw in col_lower for kw in ["_id", "id_", "code", "key", "pk", "fk"]):
            if unique_ratio >= _MIN_UNIQUE_RATIO_FOR_ID:
                return _SDTYPE_ID, {}

        # High-cardinality text
        if unique_ratio > 0.8 and unique_count > 100:
            return _SDTYPE_TEXT, {}

        # Default: categorical
        return _SDTYPE_CATEGORICAL, {}

    return _SDTYPE_CATEGORICAL, {}


def _build_column_profile(
    series: pd.Series,
    sdtype: str,
    null_fraction: float,
    unique_count: int,
) -> dict[str, Any]:
    """Build a distribution summary profile for a column.

    Args:
        series: Column data series.
        sdtype: Detected SDV semantic type.
        null_fraction: Fraction of null values.
        unique_count: Number of unique non-null values.

    Returns:
        Profile dict with distribution statistics.
    """
    profile: dict[str, Any] = {
        "null_fraction": round(float(null_fraction), 4),
        "unique_count": int(unique_count),
        "total_rows": len(series),
    }

    clean = series.dropna()

    if sdtype == _SDTYPE_NUMERICAL and len(clean) > 0:
        numeric_clean = pd.to_numeric(clean, errors="coerce").dropna()
        if len(numeric_clean) > 0:
            profile.update(
                {
                    "mean": round(float(numeric_clean.mean()), 6),
                    "std": round(float(numeric_clean.std()), 6),
                    "min": round(float(numeric_clean.min()), 6),
                    "max": round(float(numeric_clean.max()), 6),
                    "q25": round(float(numeric_clean.quantile(0.25)), 6),
                    "median": round(float(numeric_clean.quantile(0.50)), 6),
                    "q75": round(float(numeric_clean.quantile(0.75)), 6),
                    "skewness": round(float(numeric_clean.skew()), 4),
                }
            )

    elif sdtype == _SDTYPE_CATEGORICAL and len(clean) > 0:
        value_counts = clean.value_counts(normalize=True)
        profile["top_values"] = [
            {"value": str(val), "frequency": round(float(freq), 4)}
            for val, freq in value_counts.head(20).items()
        ]
        profile["entropy"] = round(float(_compute_entropy(value_counts.values)), 4)

    elif sdtype == _SDTYPE_DATETIME and len(clean) > 0:
        try:
            dt_series = pd.to_datetime(clean, errors="coerce").dropna()
            if len(dt_series) > 0:
                profile["min_date"] = str(dt_series.min())
                profile["max_date"] = str(dt_series.max())
                profile["date_range_days"] = int((dt_series.max() - dt_series.min()).days)
        except Exception:
            pass

    return profile


def _infer_column_constraints(
    col_name: str,
    series: pd.Series,
    sdtype: str,
    null_fraction: float,
) -> list[dict[str, Any]]:
    """Infer business rule constraints from column data.

    Args:
        col_name: Column name.
        series: Column data series.
        sdtype: Detected SDV semantic type.
        null_fraction: Fraction of null values.

    Returns:
        List of constraint dicts with column, constraint_type, params.
    """
    constraints: list[dict[str, Any]] = []
    clean = series.dropna()

    # Not-null constraint
    if null_fraction == 0.0:
        constraints.append(
            {"column": col_name, "constraint_type": "not_null", "params": {}}
        )

    # Range constraint for numeric columns
    if sdtype == _SDTYPE_NUMERICAL and len(clean) > 0:
        numeric_clean = pd.to_numeric(clean, errors="coerce").dropna()
        if len(numeric_clean) > 0:
            # Add small buffer to avoid over-constraining
            value_range = float(numeric_clean.max()) - float(numeric_clean.min())
            buffer = max(value_range * 0.01, 1.0)
            constraints.append(
                {
                    "column": col_name,
                    "constraint_type": "range",
                    "params": {
                        "min": round(float(numeric_clean.min()) - buffer, 4),
                        "max": round(float(numeric_clean.max()) + buffer, 4),
                    },
                }
            )

    return constraints


def _compute_entropy(probabilities: "np.ndarray") -> float:
    """Compute Shannon entropy of a probability distribution.

    Args:
        probabilities: Array of probabilities (should sum to ~1.0).

    Returns:
        Shannon entropy in nats.
    """
    probs = probabilities[probabilities > 0]
    return float(-np.sum(probs * np.log(probs + 1e-10)))

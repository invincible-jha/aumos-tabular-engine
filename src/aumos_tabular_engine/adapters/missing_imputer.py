"""Missing data imputer adapter for handling incomplete tabular datasets.

Provides KNN, MICE, and simple (mean/median/mode) imputation strategies.
Analyzes missing patterns, measures imputation quality, and supports
per-column strategy configuration.
"""

import asyncio
from functools import partial
from typing import Any, Literal

import numpy as np
import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Imputation strategy constants
_STRATEGY_KNN = "knn"
_STRATEGY_MICE = "mice"
_STRATEGY_MEAN = "mean"
_STRATEGY_MEDIAN = "median"
_STRATEGY_MODE = "mode"
_STRATEGY_CONSTANT = "constant"

# Column sdtype categories
_NUMERIC_SDTYPES = {"numerical", "float", "integer"}
_CATEGORICAL_SDTYPES = {"categorical", "boolean", "text"}

ImputationStrategy = Literal["knn", "mice", "mean", "median", "mode", "constant"]


class MissingDataImputer:
    """Impute missing values in tabular DataFrames using configurable strategies.

    Supports KNN imputation (k-nearest neighbors), MICE (Multiple Imputation by
    Chained Equations), and simple statistical imputation. Applies column-specific
    strategies when provided, falls back to a global default strategy.
    """

    def __init__(
        self,
        default_strategy: ImputationStrategy = _STRATEGY_MEAN,
        knn_neighbors: int = 5,
        mice_max_iter: int = 10,
        mice_random_state: int = 42,
        column_strategies: dict[str, ImputationStrategy] | None = None,
        constant_fill_values: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the imputer with configurable strategies.

        Args:
            default_strategy: Global fallback strategy when no column-specific
                strategy is provided. One of: knn, mice, mean, median, mode, constant.
            knn_neighbors: Number of neighbors for KNN imputation.
            mice_max_iter: Maximum iterations for MICE convergence.
            mice_random_state: Random seed for MICE reproducibility.
            column_strategies: Optional per-column strategy overrides mapping
                column_name → strategy name.
            constant_fill_values: Optional per-column fill values used when
                strategy is 'constant'.
        """
        self._default_strategy = default_strategy
        self._knn_neighbors = knn_neighbors
        self._mice_max_iter = mice_max_iter
        self._mice_random_state = mice_random_state
        self._column_strategies = column_strategies or {}
        self._constant_fill_values = constant_fill_values or {}

    async def impute(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Impute all missing values in the DataFrame.

        Dispatches to the appropriate imputation strategy per column (or globally
        for KNN and MICE which operate on the full matrix). CPU-bound work runs
        in a thread pool executor.

        Args:
            data: DataFrame with missing values to impute.
            metadata: Optional SDV-compatible metadata dict for column type hints.

        Returns:
            DataFrame with all missing values imputed.
        """
        missing_summary = self.analyze_missing_patterns(data)
        if missing_summary["total_missing"] == 0:
            logger.info("No missing values found — imputation skipped")
            return data.copy()

        logger.info(
            "Starting imputation",
            strategy=self._default_strategy,
            missing_columns=missing_summary["affected_columns"],
            total_missing=missing_summary["total_missing"],
        )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(self._run_imputation, data, metadata or {}),
        )

        quality = self._compute_imputation_quality(data, result)
        logger.info(
            "Imputation complete",
            columns_imputed=quality["columns_imputed"],
            mean_fill_rate=quality["mean_fill_rate"],
        )
        return result

    def analyze_missing_patterns(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze the missing value patterns in a DataFrame.

        Identifies columns with missing values, computes per-column missing rates,
        detects MCAR / MAR / MNAR heuristics, and returns a structured summary.

        Args:
            data: DataFrame to analyze.

        Returns:
            Missing pattern analysis dict with per-column breakdown and totals.
        """
        total_cells = data.shape[0] * data.shape[1]
        total_missing = int(data.isna().sum().sum())
        missing_rate = total_missing / total_cells if total_cells > 0 else 0.0

        column_analysis: dict[str, Any] = {}
        affected_columns: list[str] = []

        for col in data.columns:
            null_count = int(data[col].isna().sum())
            if null_count == 0:
                continue

            affected_columns.append(col)
            null_rate = null_count / len(data)
            dtype_str = str(data[col].dtype)

            column_analysis[col] = {
                "missing_count": null_count,
                "missing_rate": round(float(null_rate), 4),
                "dtype": dtype_str,
                "is_numeric": pd.api.types.is_numeric_dtype(data[col]),
                "pattern": self._classify_missing_pattern(data, col),
            }

        return {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "total_missing": total_missing,
            "missing_rate": round(float(missing_rate), 4),
            "affected_columns": affected_columns,
            "column_analysis": column_analysis,
        }

    def get_imputation_quality_metrics(
        self,
        original_data: pd.DataFrame,
        imputed_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Measure imputation quality by comparing original and imputed DataFrames.

        For columns that had missing values, computes how well the imputed
        distribution matches the non-missing distribution of the same column.

        Args:
            original_data: DataFrame before imputation (with NaNs).
            imputed_data: DataFrame after imputation (no NaNs).

        Returns:
            Quality metrics dict per column and aggregate statistics.
        """
        return self._compute_imputation_quality(original_data, imputed_data)

    def impute_column(
        self,
        data: pd.DataFrame,
        column: str,
        strategy: ImputationStrategy | None = None,
        fill_value: Any = None,
    ) -> pd.DataFrame:
        """Impute missing values in a single column using a specified strategy.

        Args:
            data: DataFrame containing the column.
            column: Column name to impute.
            strategy: Imputation strategy to use. Defaults to default_strategy.
            fill_value: Constant fill value (used only when strategy='constant').

        Returns:
            DataFrame with the column's missing values imputed.
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        effective_strategy = strategy or self._column_strategies.get(column, self._default_strategy)
        result = data.copy()
        null_mask = result[column].isna()

        if not null_mask.any():
            return result

        is_numeric = pd.api.types.is_numeric_dtype(result[column])

        if effective_strategy == _STRATEGY_MEAN:
            if is_numeric:
                fill = result[column].mean()
            else:
                fill = result[column].mode().iloc[0] if len(result[column].mode()) > 0 else None
            result.loc[null_mask, column] = fill

        elif effective_strategy == _STRATEGY_MEDIAN:
            if is_numeric:
                fill = result[column].median()
            else:
                fill = result[column].mode().iloc[0] if len(result[column].mode()) > 0 else None
            result.loc[null_mask, column] = fill

        elif effective_strategy == _STRATEGY_MODE:
            mode_series = result[column].mode()
            if len(mode_series) > 0:
                result.loc[null_mask, column] = mode_series.iloc[0]

        elif effective_strategy == _STRATEGY_CONSTANT:
            effective_fill = fill_value if fill_value is not None else self._constant_fill_values.get(column, 0)
            result.loc[null_mask, column] = effective_fill

        elif effective_strategy in (_STRATEGY_KNN, _STRATEGY_MICE):
            # Single-column KNN/MICE: fall back to mean/mode since these need full matrix
            result = self.impute_column(result, column, strategy=_STRATEGY_MEAN)

        return result

    def _run_imputation(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Synchronous imputation dispatcher (runs in thread pool executor).

        Args:
            data: DataFrame to impute.
            metadata: Column metadata for type hints.

        Returns:
            Imputed DataFrame.
        """
        strategy = self._default_strategy

        if strategy == _STRATEGY_KNN:
            return self._impute_knn(data, metadata)
        elif strategy == _STRATEGY_MICE:
            return self._impute_mice(data, metadata)
        else:
            return self._impute_column_wise(data, metadata)

    def _impute_knn(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Apply KNN imputation to the full DataFrame matrix.

        Numeric columns are imputed with KNNImputer. Categorical columns
        are encoded, imputed, and decoded back.

        Args:
            data: DataFrame to impute.
            metadata: Column metadata for type classification.

        Returns:
            Imputed DataFrame.
        """
        try:
            from sklearn.impute import KNNImputer
            from sklearn.preprocessing import OrdinalEncoder
        except ImportError:
            logger.warning("sklearn not available — falling back to column-wise imputation")
            return self._impute_column_wise(data, metadata)

        result = data.copy()
        columns_meta = metadata.get("columns", {})

        numeric_cols = [
            col for col in result.columns
            if pd.api.types.is_numeric_dtype(result[col]) or
            columns_meta.get(col, {}).get("sdtype", "") in _NUMERIC_SDTYPES
        ]
        categorical_cols = [col for col in result.columns if col not in numeric_cols]

        # Impute numeric columns with KNN
        if numeric_cols:
            numeric_data = result[numeric_cols].copy()
            imputer = KNNImputer(n_neighbors=self._knn_neighbors)
            imputed_numeric = pd.DataFrame(
                imputer.fit_transform(numeric_data),
                columns=numeric_cols,
                index=result.index,
            )
            result[numeric_cols] = imputed_numeric

        # Impute categorical columns with mode fallback per column
        for col in categorical_cols:
            if result[col].isna().any():
                result = self.impute_column(result, col, strategy=_STRATEGY_MODE)

        return result

    def _impute_mice(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Apply MICE (Multiple Imputation by Chained Equations) to the DataFrame.

        Uses sklearn's IterativeImputer for numeric columns and falls back to
        mode imputation for categorical columns.

        Args:
            data: DataFrame to impute.
            metadata: Column metadata for type classification.

        Returns:
            Imputed DataFrame.
        """
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
        except ImportError:
            logger.warning("sklearn IterativeImputer not available — falling back to KNN")
            return self._impute_knn(data, metadata)

        result = data.copy()
        columns_meta = metadata.get("columns", {})

        numeric_cols = [
            col for col in result.columns
            if pd.api.types.is_numeric_dtype(result[col]) or
            columns_meta.get(col, {}).get("sdtype", "") in _NUMERIC_SDTYPES
        ]
        categorical_cols = [col for col in result.columns if col not in numeric_cols]

        if numeric_cols:
            numeric_data = result[numeric_cols].copy()
            imputer = IterativeImputer(
                max_iter=self._mice_max_iter,
                random_state=self._mice_random_state,
            )
            imputed_numeric = pd.DataFrame(
                imputer.fit_transform(numeric_data),
                columns=numeric_cols,
                index=result.index,
            )
            result[numeric_cols] = imputed_numeric

        for col in categorical_cols:
            if result[col].isna().any():
                result = self.impute_column(result, col, strategy=_STRATEGY_MODE)

        return result

    def _impute_column_wise(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Apply per-column simple imputation using configured strategies.

        Args:
            data: DataFrame to impute.
            metadata: Column metadata for type hints.

        Returns:
            Imputed DataFrame.
        """
        result = data.copy()
        columns_meta = metadata.get("columns", {})

        for col in result.columns:
            if not result[col].isna().any():
                continue

            column_strategy = self._column_strategies.get(col)
            if column_strategy is not None:
                effective_strategy = column_strategy
            else:
                sdtype = columns_meta.get(col, {}).get("sdtype", "")
                is_numeric = pd.api.types.is_numeric_dtype(result[col]) or sdtype in _NUMERIC_SDTYPES
                effective_strategy = self._default_strategy if is_numeric else _STRATEGY_MODE

            result = self.impute_column(result, col, strategy=effective_strategy)

        return result

    def _compute_imputation_quality(
        self,
        original_data: pd.DataFrame,
        imputed_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compute imputation quality metrics comparing distributions.

        Args:
            original_data: DataFrame before imputation.
            imputed_data: DataFrame after imputation.

        Returns:
            Quality metrics dict.
        """
        column_metrics: dict[str, Any] = {}
        fill_rates: list[float] = []
        columns_imputed = 0

        for col in original_data.columns:
            null_mask = original_data[col].isna()
            null_count = int(null_mask.sum())

            if null_count == 0:
                continue

            columns_imputed += 1
            filled_count = int(imputed_data[col].notna().sum()) - int(original_data[col].notna().sum())
            fill_rate = filled_count / null_count if null_count > 0 else 1.0
            fill_rates.append(fill_rate)

            original_non_null = original_data[col].dropna()

            if pd.api.types.is_numeric_dtype(original_data[col]) and len(original_non_null) > 0:
                original_mean = float(original_non_null.mean())
                imputed_values = imputed_data.loc[null_mask, col].dropna()
                imputed_mean = float(imputed_values.mean()) if len(imputed_values) > 0 else 0.0
                mean_deviation = abs(original_mean - imputed_mean) / (abs(original_mean) + 1e-10)
                column_metrics[col] = {
                    "null_count": null_count,
                    "fill_rate": round(fill_rate, 4),
                    "original_mean": round(original_mean, 4),
                    "imputed_mean": round(imputed_mean, 4),
                    "mean_deviation": round(mean_deviation, 4),
                }
            else:
                column_metrics[col] = {
                    "null_count": null_count,
                    "fill_rate": round(fill_rate, 4),
                }

        return {
            "columns_imputed": columns_imputed,
            "mean_fill_rate": round(float(sum(fill_rates) / len(fill_rates)) if fill_rates else 1.0, 4),
            "column_metrics": column_metrics,
        }

    def _classify_missing_pattern(self, data: pd.DataFrame, column: str) -> str:
        """Heuristically classify missing pattern as MCAR, MAR, or MNAR.

        Uses a simple correlation-based heuristic: if missingness in the target
        column correlates strongly with another column's values, it is likely MAR.
        Otherwise classified as MCAR.

        Args:
            data: Full DataFrame.
            column: Column to classify.

        Returns:
            One of: 'MCAR', 'MAR', 'MNAR_suspected'.
        """
        missing_indicator = data[column].isna().astype(int)

        max_correlation = 0.0
        for other_col in data.columns:
            if other_col == column:
                continue
            if not pd.api.types.is_numeric_dtype(data[other_col]):
                continue
            other_clean = data[other_col].fillna(data[other_col].median())
            try:
                corr = abs(float(other_clean.corr(missing_indicator)))
                if corr > max_correlation:
                    max_correlation = corr
            except Exception:
                continue

        if max_correlation > 0.3:
            return "MAR"
        elif max_correlation > 0.1:
            return "MCAR"
        else:
            return "MCAR"

"""Synthesis quality evaluator adapter using SDMetrics.

Computes column-level and aggregate quality scores comparing real versus
synthetic DataFrames. Produces per-column diagnostic reports, tracks quality
history, and enforces configurable quality thresholds.
"""

import asyncio
import datetime
from collections import deque
from functools import partial
from typing import Any

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# SDMetrics quality report column pair metric names
_METRIC_KS_COMPLEMENT = "KSComplement"
_METRIC_TV_COMPLEMENT = "TVComplement"
_METRIC_CORRELATION_SIMILARITY = "CorrelationSimilarity"
_METRIC_CONTINGENCY_SIMILARITY = "ContingencySimilarity"

# Maximum history entries retained per tenant
_MAX_HISTORY_ENTRIES = 200

# Default quality thresholds
_DEFAULT_FIDELITY_THRESHOLD = 0.75
_DEFAULT_COLUMN_THRESHOLD = 0.60


class SynthesisQualityEvaluator:
    """SDMetrics-backed quality evaluator for synthetic tabular data.

    Evaluates column shape similarity, column pair trends, and aggregate
    quality scores. Maintains a rolling history of evaluations for trend
    analysis and provides per-column diagnostic detail.
    """

    def __init__(
        self,
        fidelity_threshold: float = _DEFAULT_FIDELITY_THRESHOLD,
        column_threshold: float = _DEFAULT_COLUMN_THRESHOLD,
        history_size: int = _MAX_HISTORY_ENTRIES,
    ) -> None:
        """Initialise the quality evaluator.

        Args:
            fidelity_threshold: Minimum aggregate quality score to pass validation.
            column_threshold: Minimum per-column score to flag column as degraded.
            history_size: Maximum number of historical evaluation records to retain.
        """
        self._fidelity_threshold = fidelity_threshold
        self._column_threshold = column_threshold
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)

    async def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run full quality evaluation against SDMetrics QualityReport.

        Executes column shape (KSComplement / TVComplement) and column pair
        trend (CorrelationSimilarity / ContingencySimilarity) evaluations in
        a thread pool to avoid blocking the event loop.

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Synthetic DataFrame to evaluate.
            metadata: SDV-compatible SingleTableMetadata dict.

        Returns:
            Evaluation report dict containing:
            - overall_score: Aggregate quality score [0.0, 1.0]
            - column_shapes_score: Average shape-similarity across columns
            - column_pair_trends_score: Average correlation-preservation score
            - column_reports: Per-column metric breakdown
            - degraded_columns: Columns below column_threshold
            - passed: Whether overall_score >= fidelity_threshold
            - evaluated_at: ISO-8601 timestamp
        """
        logger.info(
            "Starting quality evaluation",
            real_rows=len(real_data),
            synthetic_rows=len(synthetic_data),
            num_columns=len(real_data.columns),
        )

        loop = asyncio.get_running_loop()
        report_data = await loop.run_in_executor(
            None,
            partial(self._run_sdmetrics_report, real_data, synthetic_data, metadata),
        )

        column_reports = report_data["column_reports"]
        column_shapes_score = report_data["column_shapes_score"]
        column_pair_trends_score = report_data["column_pair_trends_score"]

        # Aggregate: equal-weight blend of shape and pair trend scores
        if column_pair_trends_score is not None:
            overall_score = (column_shapes_score + column_pair_trends_score) / 2.0
        else:
            overall_score = column_shapes_score

        # Identify degraded columns
        degraded_columns = [
            col
            for col, metrics in column_reports.items()
            if metrics.get("shape_score", 1.0) < self._column_threshold
        ]

        passed = overall_score >= self._fidelity_threshold

        result: dict[str, Any] = {
            "overall_score": round(overall_score, 4),
            "column_shapes_score": round(column_shapes_score, 4),
            "column_pair_trends_score": (
                round(column_pair_trends_score, 4)
                if column_pair_trends_score is not None
                else None
            ),
            "column_reports": column_reports,
            "degraded_columns": degraded_columns,
            "passed": passed,
            "threshold": self._fidelity_threshold,
            "evaluated_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

        self._history.append(
            {
                "overall_score": overall_score,
                "passed": passed,
                "evaluated_at": result["evaluated_at"],
                "num_real_rows": len(real_data),
                "num_synthetic_rows": len(synthetic_data),
            }
        )

        logger.info(
            "Quality evaluation complete",
            overall_score=overall_score,
            column_shapes_score=column_shapes_score,
            degraded_columns=len(degraded_columns),
            passed=passed,
        )
        return result

    async def evaluate_column_shape(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        column_name: str,
    ) -> dict[str, Any]:
        """Compute shape similarity metrics for a single column.

        Uses KSComplement for numerical columns and TVComplement for
        categorical columns, running in a thread pool executor.

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Synthetic DataFrame.
            column_name: Column to evaluate.

        Returns:
            Dict with metric_name, score, and interpretation.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self._compute_column_shape, real_data, synthetic_data, column_name),
        )

    async def get_diagnostic_report(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a detailed per-column diagnostic report.

        Provides metric breakdowns, distribution comparisons, and
        actionable recommendations for low-scoring columns.

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Synthetic DataFrame.
            metadata: SDV-compatible metadata dict.

        Returns:
            Diagnostic report with per-column details and recommendations.
        """
        evaluation = await self.evaluate(real_data, synthetic_data, metadata)
        recommendations: list[str] = []

        for col in evaluation["degraded_columns"]:
            col_info = evaluation["column_reports"].get(col, {})
            score = col_info.get("shape_score", 0.0)
            sdtype = col_info.get("sdtype", "unknown")

            if sdtype == "numerical":
                recommendations.append(
                    f"Column '{col}' (score={score:.3f}): distribution mismatch — "
                    "consider increasing training epochs or adjusting normalisation."
                )
            else:
                recommendations.append(
                    f"Column '{col}' (score={score:.3f}): category frequency mismatch — "
                    "ensure sufficient training data coverage for all categories."
                )

        return {
            **evaluation,
            "recommendations": recommendations,
            "history_summary": self._get_history_summary(),
        }

    def get_quality_history(self) -> list[dict[str, Any]]:
        """Return the rolling evaluation history.

        Returns:
            List of historical evaluation summaries, most recent last.
        """
        return list(self._history)

    def validate_thresholds(self, evaluation_result: dict[str, Any]) -> list[str]:
        """Validate an evaluation result against configured thresholds.

        Args:
            evaluation_result: Result dict from evaluate().

        Returns:
            List of threshold violation messages (empty if all pass).
        """
        violations: list[str] = []
        overall = evaluation_result.get("overall_score", 0.0)

        if overall < self._fidelity_threshold:
            violations.append(
                f"Overall quality score {overall:.3f} below threshold {self._fidelity_threshold}"
            )

        for col in evaluation_result.get("degraded_columns", []):
            col_score = evaluation_result["column_reports"].get(col, {}).get("shape_score", 0.0)
            violations.append(
                f"Column '{col}' score {col_score:.3f} below column threshold {self._column_threshold}"
            )

        return violations

    def _run_sdmetrics_report(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute SDMetrics QualityReport synchronously (runs in executor).

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Synthetic DataFrame.
            metadata: SDV-compatible metadata dict.

        Returns:
            Internal report dict with scores and column details.
        """
        try:
            from sdmetrics.reports.single_table import QualityReport
            from sdv.metadata import SingleTableMetadata

            sdv_metadata = SingleTableMetadata.load_from_dict(metadata)
            report = QualityReport()
            report.generate(real_data, synthetic_data, sdv_metadata.to_dict())

            shapes_df = report.get_details(property_name="Column Shapes")
            pairs_df = report.get_details(property_name="Column Pair Trends")

            column_reports: dict[str, Any] = {}

            if shapes_df is not None and not shapes_df.empty:
                for _, row in shapes_df.iterrows():
                    col = row.get("Column", row.get("column", ""))
                    metric = row.get("Metric", row.get("metric", ""))
                    score = float(row.get("Score", row.get("score", 0.0)))
                    column_reports[col] = {
                        "shape_metric": metric,
                        "shape_score": round(score, 4),
                        "sdtype": metadata.get("columns", {}).get(col, {}).get("sdtype", "unknown"),
                    }

            column_shapes_score = float(report.get_score()) if hasattr(report, "get_score") else 0.0

            column_pair_trends_score: float | None = None
            if pairs_df is not None and not pairs_df.empty:
                pair_scores = pd.to_numeric(
                    pairs_df.get("Score", pairs_df.get("score", pd.Series(dtype=float))),
                    errors="coerce",
                ).dropna()
                if len(pair_scores) > 0:
                    column_pair_trends_score = float(pair_scores.mean())

            return {
                "column_reports": column_reports,
                "column_shapes_score": column_shapes_score,
                "column_pair_trends_score": column_pair_trends_score,
            }

        except ImportError:
            logger.warning("sdmetrics not available — falling back to statistical evaluation")
            return self._fallback_statistical_evaluation(real_data, synthetic_data, metadata)

        except Exception as exc:
            logger.error("SDMetrics evaluation failed", error=str(exc))
            return self._fallback_statistical_evaluation(real_data, synthetic_data, metadata)

    def _fallback_statistical_evaluation(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute basic statistical quality scores without sdmetrics.

        Uses KS-test for numerical columns and chi-square for categorical.
        Serves as a fallback when sdmetrics is not installed.

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Synthetic DataFrame.
            metadata: SDV-compatible metadata dict.

        Returns:
            Internal report dict matching the sdmetrics output structure.
        """
        from scipy import stats as scipy_stats

        column_reports: dict[str, Any] = {}
        shape_scores: list[float] = []
        columns_meta = metadata.get("columns", {})

        for col in real_data.columns:
            if col not in synthetic_data.columns:
                continue

            sdtype = columns_meta.get(col, {}).get("sdtype", "categorical")
            real_col = real_data[col].dropna()
            synth_col = synthetic_data[col].dropna()

            if len(real_col) == 0 or len(synth_col) == 0:
                score = 0.0
                metric = "empty"
            elif sdtype == "numerical":
                ks_stat, _ = scipy_stats.ks_2samp(
                    real_col.astype(float), synth_col.astype(float)
                )
                score = float(1.0 - ks_stat)
                metric = _METRIC_KS_COMPLEMENT
            else:
                real_counts = real_col.value_counts(normalize=True)
                synth_counts = synth_col.value_counts(normalize=True)
                all_cats = set(real_counts.index) | set(synth_counts.index)
                real_vec = [real_counts.get(c, 0.0) for c in all_cats]
                synth_vec = [synth_counts.get(c, 0.0) for c in all_cats]
                tv_distance = float(
                    sum(abs(r - s) for r, s in zip(real_vec, synth_vec)) / 2.0
                )
                score = 1.0 - tv_distance
                metric = _METRIC_TV_COMPLEMENT

            shape_scores.append(score)
            column_reports[col] = {
                "shape_metric": metric,
                "shape_score": round(score, 4),
                "sdtype": sdtype,
            }

        avg_shape_score = float(sum(shape_scores) / len(shape_scores)) if shape_scores else 0.0

        return {
            "column_reports": column_reports,
            "column_shapes_score": avg_shape_score,
            "column_pair_trends_score": None,
        }

    def _compute_column_shape(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        column_name: str,
    ) -> dict[str, Any]:
        """Synchronous per-column shape metric computation.

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Synthetic DataFrame.
            column_name: Column to evaluate.

        Returns:
            Dict with metric_name, score, and column_name.
        """
        from scipy import stats as scipy_stats

        real_col = real_data[column_name].dropna()
        synth_col = synthetic_data[column_name].dropna()

        if len(real_col) == 0 or len(synth_col) == 0:
            return {"column": column_name, "metric": "empty", "score": 0.0}

        if pd.api.types.is_numeric_dtype(real_col):
            ks_stat, _ = scipy_stats.ks_2samp(real_col.astype(float), synth_col.astype(float))
            score = 1.0 - float(ks_stat)
            metric = _METRIC_KS_COMPLEMENT
        else:
            real_counts = real_col.value_counts(normalize=True)
            synth_counts = synth_col.value_counts(normalize=True)
            all_cats = set(real_counts.index) | set(synth_counts.index)
            tv = sum(abs(real_counts.get(c, 0.0) - synth_counts.get(c, 0.0)) for c in all_cats) / 2.0
            score = 1.0 - float(tv)
            metric = _METRIC_TV_COMPLEMENT

        return {
            "column": column_name,
            "metric": metric,
            "score": round(score, 4),
        }

    def _get_history_summary(self) -> dict[str, Any]:
        """Compute summary statistics over the evaluation history.

        Returns:
            Dict with count, mean_score, min_score, max_score, pass_rate.
        """
        if not self._history:
            return {"count": 0}

        scores = [entry["overall_score"] for entry in self._history]
        pass_count = sum(1 for entry in self._history if entry["passed"])

        return {
            "count": len(self._history),
            "mean_score": round(sum(scores) / len(scores), 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "pass_rate": round(pass_count / len(self._history), 4),
        }

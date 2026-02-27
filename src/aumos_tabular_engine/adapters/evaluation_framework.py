"""Evaluation framework adapter for fidelity-privacy tradeoff analysis.

Computes Pareto frontiers across generator configurations, runs multi-objective
optimization metrics, produces benchmark comparisons, and generates structured
evaluation reports for synthetic dataset quality assessment.
"""

import asyncio
import datetime
import uuid
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Generator type labels used in benchmark comparisons
_KNOWN_GENERATOR_TYPES = frozenset({"ctgan", "tvae", "gaussian_copula", "smartnoise", "hma"})

# Objective weight presets for multi-objective scoring
_WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "balanced": {"fidelity": 0.5, "privacy": 0.5},
    "fidelity_first": {"fidelity": 0.8, "privacy": 0.2},
    "privacy_first": {"fidelity": 0.2, "privacy": 0.8},
    "equal_three": {"fidelity": 0.33, "privacy": 0.33, "utility": 0.34},
}


@dataclass
class EvaluationPoint:
    """A single evaluation result point in the tradeoff space.

    Attributes:
        generator_type: Generator backend used (ctgan, tvae, etc.).
        epsilon: Differential privacy epsilon (inf if no DP used).
        fidelity_score: SDMetrics quality score [0.0, 1.0].
        privacy_score: Anonymeter/DP privacy score [0.0, 1.0].
        utility_score: Optional downstream utility score [0.0, 1.0].
        training_time_s: Training duration in seconds.
        generation_time_s: Generation duration in seconds.
        num_rows: Number of synthetic rows generated.
        run_id: Optional UUID identifying this evaluation run.
        metadata: Additional key-value context for this point.
    """

    generator_type: str
    epsilon: float
    fidelity_score: float
    privacy_score: float
    utility_score: float = 0.0
    training_time_s: float = 0.0
    generation_time_s: float = 0.0
    num_rows: int = 0
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON serialization.

        Returns:
            Dict representation of this evaluation point.
        """
        return {
            "generator_type": self.generator_type,
            "epsilon": self.epsilon,
            "fidelity_score": round(self.fidelity_score, 4),
            "privacy_score": round(self.privacy_score, 4),
            "utility_score": round(self.utility_score, 4),
            "training_time_s": round(self.training_time_s, 3),
            "generation_time_s": round(self.generation_time_s, 3),
            "num_rows": self.num_rows,
            "run_id": self.run_id,
            "metadata": self.metadata,
        }


class EvaluationFramework:
    """Multi-objective evaluation framework for synthetic data quality assessment.

    Computes Pareto frontiers over fidelity vs privacy tradeoff curves,
    performs benchmark comparisons across generator types, recommends optimal
    configurations, and generates structured evaluation reports.
    """

    def __init__(
        self,
        fidelity_weight: float = 0.5,
        privacy_weight: float = 0.5,
        utility_weight: float = 0.0,
        min_fidelity_threshold: float = 0.70,
        min_privacy_threshold: float = 0.50,
    ) -> None:
        """Initialise the evaluation framework.

        Args:
            fidelity_weight: Weight for fidelity in composite scoring [0, 1].
            privacy_weight: Weight for privacy in composite scoring [0, 1].
            utility_weight: Weight for utility in composite scoring [0, 1].
                Weights need not sum to 1.0; they are normalized internally.
            min_fidelity_threshold: Minimum fidelity for a point to be considered
                production-ready.
            min_privacy_threshold: Minimum privacy score for a point to be
                considered production-ready.
        """
        total_weight = fidelity_weight + privacy_weight + utility_weight
        if total_weight <= 0:
            raise ValueError("At least one objective weight must be positive")

        self._fidelity_weight = fidelity_weight / total_weight
        self._privacy_weight = privacy_weight / total_weight
        self._utility_weight = utility_weight / total_weight
        self._min_fidelity = min_fidelity_threshold
        self._min_privacy = min_privacy_threshold
        self._evaluation_history: list[EvaluationPoint] = []

    def record_evaluation(self, point: EvaluationPoint) -> None:
        """Add an evaluation point to the historical record.

        Args:
            point: EvaluationPoint to record.
        """
        self._evaluation_history.append(point)
        logger.info(
            "Evaluation point recorded",
            generator_type=point.generator_type,
            epsilon=point.epsilon,
            fidelity=point.fidelity_score,
            privacy=point.privacy_score,
        )

    async def compute_pareto_frontier(
        self,
        points: list[EvaluationPoint] | None = None,
    ) -> list[EvaluationPoint]:
        """Compute Pareto-optimal points in the fidelity-privacy space.

        A point is Pareto-optimal if no other point dominates it — i.e., no
        other point is simultaneously better in all objective dimensions.

        Args:
            points: Evaluation points to analyze. Defaults to the historical record.

        Returns:
            List of Pareto-optimal EvaluationPoints sorted by epsilon.
        """
        candidate_points = points if points is not None else self._evaluation_history

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self._compute_pareto_sync, candidate_points),
        )

    async def recommend_optimal_config(
        self,
        points: list[EvaluationPoint] | None = None,
        require_production_ready: bool = True,
    ) -> dict[str, Any]:
        """Recommend the optimal generator configuration from evaluated points.

        Selects the point with the highest composite weighted score among
        production-ready points (above fidelity and privacy thresholds).

        Args:
            points: Evaluation points to consider. Defaults to historical record.
            require_production_ready: If True, only consider points above thresholds.

        Returns:
            Recommendation dict with the best EvaluationPoint and scoring details.
        """
        candidate_points = points if points is not None else self._evaluation_history

        if not candidate_points:
            return {
                "recommendation": None,
                "reason": "No evaluation points available",
                "candidate_count": 0,
            }

        if require_production_ready:
            production_ready = [
                p for p in candidate_points
                if p.fidelity_score >= self._min_fidelity
                and p.privacy_score >= self._min_privacy
            ]
        else:
            production_ready = list(candidate_points)

        if not production_ready:
            return {
                "recommendation": None,
                "reason": (
                    f"No production-ready points found above thresholds "
                    f"(fidelity>={self._min_fidelity}, privacy>={self._min_privacy})"
                ),
                "candidate_count": len(candidate_points),
                "production_ready_count": 0,
            }

        scored_points = [
            (p, self._composite_score(p))
            for p in production_ready
        ]
        scored_points.sort(key=lambda x: x[1], reverse=True)

        best_point, best_score = scored_points[0]

        return {
            "recommendation": best_point.to_dict(),
            "composite_score": round(best_score, 4),
            "reason": "highest weighted composite of fidelity, privacy, and utility",
            "weights": {
                "fidelity": round(self._fidelity_weight, 3),
                "privacy": round(self._privacy_weight, 3),
                "utility": round(self._utility_weight, 3),
            },
            "candidate_count": len(candidate_points),
            "production_ready_count": len(production_ready),
        }

    async def benchmark_generators(
        self,
        points: list[EvaluationPoint] | None = None,
    ) -> dict[str, Any]:
        """Compare generator types across all evaluation dimensions.

        Aggregates fidelity, privacy, utility, and timing statistics per
        generator type and produces a ranked comparison table.

        Args:
            points: Evaluation points. Defaults to the historical record.

        Returns:
            Benchmark comparison dict with per-generator summaries and rankings.
        """
        candidate_points = points if points is not None else self._evaluation_history

        if not candidate_points:
            return {"generators": {}, "rankings": [], "total_evaluations": 0}

        generator_groups: dict[str, list[EvaluationPoint]] = {}
        for point in candidate_points:
            generator_groups.setdefault(point.generator_type, []).append(point)

        generator_summaries: dict[str, dict[str, Any]] = {}
        for gen_type, gen_points in generator_groups.items():
            generator_summaries[gen_type] = self._summarize_generator_points(gen_points)

        rankings = sorted(
            [
                {"generator_type": gen_type, **summary}
                for gen_type, summary in generator_summaries.items()
            ],
            key=lambda g: g.get("mean_composite_score", 0.0),
            reverse=True,
        )
        for rank, entry in enumerate(rankings, start=1):
            entry["rank"] = rank

        return {
            "generators": generator_summaries,
            "rankings": rankings,
            "total_evaluations": len(candidate_points),
            "generator_count": len(generator_groups),
        }

    async def compute_tradeoff_visualization_data(
        self,
        points: list[EvaluationPoint] | None = None,
    ) -> dict[str, Any]:
        """Produce visualization-ready data for the fidelity-privacy tradeoff.

        Formats evaluation points and Pareto frontier for charting,
        including axis ranges, series data, and annotation hints.

        Args:
            points: Evaluation points. Defaults to the historical record.

        Returns:
            Visualization data dict suitable for frontend charting libraries.
        """
        candidate_points = points if points is not None else self._evaluation_history
        pareto_points = await self.compute_pareto_frontier(candidate_points)

        all_series: list[dict[str, Any]] = [p.to_dict() for p in candidate_points]
        pareto_series: list[dict[str, Any]] = [p.to_dict() for p in pareto_points]

        fidelity_values = [p.fidelity_score for p in candidate_points]
        privacy_values = [p.privacy_score for p in candidate_points]

        return {
            "all_points": all_series,
            "pareto_frontier": pareto_series,
            "axes": {
                "x": {
                    "label": "Fidelity Score",
                    "min": round(min(fidelity_values, default=0.0), 3),
                    "max": round(max(fidelity_values, default=1.0), 3),
                },
                "y": {
                    "label": "Privacy Score",
                    "min": round(min(privacy_values, default=0.0), 3),
                    "max": round(max(privacy_values, default=1.0), 3),
                },
            },
            "thresholds": {
                "min_fidelity": self._min_fidelity,
                "min_privacy": self._min_privacy,
            },
            "total_points": len(candidate_points),
            "pareto_point_count": len(pareto_points),
        }

    async def generate_report(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        points: list[EvaluationPoint] | None = None,
        report_title: str = "Synthetic Data Evaluation Report",
    ) -> dict[str, Any]:
        """Generate a comprehensive evaluation report.

        Combines Pareto frontier analysis, benchmark comparison, optimal config
        recommendation, and dataset statistics into a single structured report.

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Generated synthetic DataFrame.
            points: Evaluation points. Defaults to the historical record.
            report_title: Human-readable report title.

        Returns:
            Full evaluation report dict.
        """
        candidate_points = points if points is not None else self._evaluation_history

        pareto_frontier, benchmark, recommendation, viz_data = await asyncio.gather(
            self.compute_pareto_frontier(candidate_points),
            self.benchmark_generators(candidate_points),
            self.recommend_optimal_config(candidate_points),
            self.compute_tradeoff_visualization_data(candidate_points),
        )

        dataset_summary = self._summarize_datasets(real_data, synthetic_data)

        report = {
            "title": report_title,
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "dataset_summary": dataset_summary,
            "evaluation_summary": {
                "total_points": len(candidate_points),
                "pareto_points": len(pareto_frontier),
                "generator_types_evaluated": list(
                    {p.generator_type for p in candidate_points}
                ),
            },
            "optimal_configuration": recommendation,
            "pareto_frontier": [p.to_dict() for p in pareto_frontier],
            "benchmark_comparison": benchmark,
            "visualization_data": viz_data,
            "thresholds": {
                "min_fidelity": self._min_fidelity,
                "min_privacy": self._min_privacy,
            },
            "objective_weights": {
                "fidelity": round(self._fidelity_weight, 3),
                "privacy": round(self._privacy_weight, 3),
                "utility": round(self._utility_weight, 3),
            },
        }

        logger.info(
            "Evaluation report generated",
            total_points=len(candidate_points),
            pareto_points=len(pareto_frontier),
            has_recommendation=recommendation.get("recommendation") is not None,
        )
        return report

    def get_evaluation_history(self) -> list[dict[str, Any]]:
        """Return all recorded evaluation points as serializable dicts.

        Returns:
            List of evaluation point dicts.
        """
        return [p.to_dict() for p in self._evaluation_history]

    def clear_history(self) -> None:
        """Clear the in-memory evaluation history."""
        self._evaluation_history.clear()
        logger.info("Evaluation history cleared")

    def _compute_pareto_sync(
        self,
        points: list[EvaluationPoint],
    ) -> list[EvaluationPoint]:
        """Synchronous Pareto frontier computation (runs in thread pool).

        A point is on the Pareto frontier if it is not dominated in all
        objective dimensions (fidelity, privacy, utility) by any other point.

        Args:
            points: All evaluation points.

        Returns:
            Pareto-optimal subset sorted by epsilon ascending.
        """
        if not points:
            return []

        pareto: list[EvaluationPoint] = []
        for candidate in points:
            dominated = False
            for other in points:
                if other is candidate:
                    continue
                if (
                    other.fidelity_score >= candidate.fidelity_score
                    and other.privacy_score >= candidate.privacy_score
                    and other.utility_score >= candidate.utility_score
                    and (
                        other.fidelity_score > candidate.fidelity_score
                        or other.privacy_score > candidate.privacy_score
                        or other.utility_score > candidate.utility_score
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(candidate)

        return sorted(pareto, key=lambda p: p.epsilon)

    def _composite_score(self, point: EvaluationPoint) -> float:
        """Compute the weighted composite objective score for a point.

        Args:
            point: Evaluation point to score.

        Returns:
            Composite score in [0.0, 1.0].
        """
        return (
            self._fidelity_weight * point.fidelity_score
            + self._privacy_weight * point.privacy_score
            + self._utility_weight * point.utility_score
        )

    def _summarize_generator_points(
        self,
        points: list[EvaluationPoint],
    ) -> dict[str, Any]:
        """Aggregate statistics for a group of same-type evaluation points.

        Args:
            points: Evaluation points for one generator type.

        Returns:
            Aggregated statistics dict.
        """
        fidelity_scores = [p.fidelity_score for p in points]
        privacy_scores = [p.privacy_score for p in points]
        training_times = [p.training_time_s for p in points]
        composite_scores = [self._composite_score(p) for p in points]

        production_ready_count = sum(
            1 for p in points
            if p.fidelity_score >= self._min_fidelity
            and p.privacy_score >= self._min_privacy
        )

        return {
            "evaluation_count": len(points),
            "production_ready_count": production_ready_count,
            "mean_fidelity": round(sum(fidelity_scores) / len(fidelity_scores), 4),
            "max_fidelity": round(max(fidelity_scores), 4),
            "min_fidelity": round(min(fidelity_scores), 4),
            "mean_privacy": round(sum(privacy_scores) / len(privacy_scores), 4),
            "max_privacy": round(max(privacy_scores), 4),
            "min_privacy": round(min(privacy_scores), 4),
            "mean_composite_score": round(
                sum(composite_scores) / len(composite_scores), 4
            ),
            "max_composite_score": round(max(composite_scores), 4),
            "mean_training_time_s": round(
                sum(training_times) / len(training_times), 2
            ) if training_times else 0.0,
            "epsilon_range": [
                round(min(p.epsilon for p in points), 4),
                round(max(p.epsilon for p in points), 4),
            ],
        }

    def _summarize_datasets(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate a dataset comparison summary.

        Args:
            real_data: Original source DataFrame.
            synthetic_data: Generated synthetic DataFrame.

        Returns:
            Dataset summary dict with row/column counts and null rate comparison.
        """
        real_null_rate = float(real_data.isna().sum().sum()) / max(real_data.size, 1)
        synth_null_rate = float(synthetic_data.isna().sum().sum()) / max(synthetic_data.size, 1)

        return {
            "real_data": {
                "row_count": len(real_data),
                "column_count": len(real_data.columns),
                "null_rate": round(real_null_rate, 4),
                "memory_mb": round(real_data.memory_usage(deep=True).sum() / 1_048_576, 2),
            },
            "synthetic_data": {
                "row_count": len(synthetic_data),
                "column_count": len(synthetic_data.columns),
                "null_rate": round(synth_null_rate, 4),
                "memory_mb": round(
                    synthetic_data.memory_usage(deep=True).sum() / 1_048_576, 2
                ),
            },
            "size_ratio": round(len(synthetic_data) / max(len(real_data), 1), 4),
            "column_overlap": len(
                set(real_data.columns) & set(synthetic_data.columns)
            ),
        }

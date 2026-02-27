"""Constraint solver adapter for referential integrity and business rule enforcement.

Enforces foreign key relationships, uniqueness constraints, value range checks,
and enum membership rules on synthetic DataFrames. Iteratively repairs violations
up to a configurable maximum attempt count.
"""

import asyncio
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Constraint type constants
_CONSTRAINT_FOREIGN_KEY = "foreign_key"
_CONSTRAINT_UNIQUE = "unique"
_CONSTRAINT_RANGE = "range"
_CONSTRAINT_ENUM = "enum"
_CONSTRAINT_NOT_NULL = "not_null"
_CONSTRAINT_REGEX = "regex"

# Maximum repair iteration rounds before giving up
_MAX_REPAIR_ITERATIONS = 5


class ConstraintSolver:
    """Enforce relational and business rule constraints on synthetic DataFrames.

    Handles foreign key lookups, uniqueness guarantees, range clamping, enum
    filtering, and not-null enforcement. Produces structured violation reports
    with per-constraint breakdowns, and iteratively repairs violations.
    """

    def __init__(
        self,
        max_repair_iterations: int = _MAX_REPAIR_ITERATIONS,
        raise_on_unresolvable: bool = False,
    ) -> None:
        """Initialise the constraint solver.

        Args:
            max_repair_iterations: Maximum repair passes before halting.
            raise_on_unresolvable: If True, raise ValueError when violations
                remain after all repair iterations. Defaults to logging a warning.
        """
        self._max_repair_iterations = max_repair_iterations
        self._raise_on_unresolvable = raise_on_unresolvable

    async def validate_async(
        self,
        data: pd.DataFrame,
        constraints: list[dict[str, Any]],
    ) -> list[str]:
        """Validate synthetic data against constraints asynchronously.

        Runs constraint checking in a thread pool to avoid blocking the event
        loop for large DataFrames.

        Args:
            data: Synthetic DataFrame to validate.
            constraints: List of constraint descriptor dicts.

        Returns:
            List of violation messages (empty if all constraints pass).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self.validate, data, constraints),
        )

    def validate(
        self,
        data: pd.DataFrame,
        constraints: list[dict[str, Any]],
    ) -> list[str]:
        """Validate synthetic data against a set of constraints synchronously.

        Args:
            data: Synthetic DataFrame to validate.
            constraints: List of constraint descriptor dicts. Each must have
                a 'constraint_type' key and optional 'column' / 'params' keys.

        Returns:
            List of human-readable violation messages (empty on full compliance).
        """
        violations: list[str] = []
        for constraint in constraints:
            constraint_type = constraint.get("constraint_type", "")
            column = constraint.get("column", "")
            params = constraint.get("params", {})

            try:
                new_violations = self._check_constraint(data, constraint_type, column, params)
                violations.extend(new_violations)
            except Exception as exc:
                violations.append(
                    f"Constraint check error [{constraint_type}] on '{column}': {exc}"
                )

        return violations

    def apply(
        self,
        data: pd.DataFrame,
        constraints: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Apply constraints to synthetic data, repairing violations iteratively.

        Runs up to max_repair_iterations passes, applying all repair transforms
        on each pass and stopping early when no violations remain.

        Args:
            data: Synthetic DataFrame to transform.
            constraints: List of constraint descriptors.

        Returns:
            Transformed DataFrame satisfying all enforceable constraints.
        """
        result = data.copy()

        for iteration in range(self._max_repair_iterations):
            violations = self.validate(result, constraints)
            if not violations:
                logger.info(
                    "All constraints satisfied",
                    iteration=iteration,
                    num_constraints=len(constraints),
                )
                return result

            logger.info(
                "Applying constraint repairs",
                iteration=iteration + 1,
                violation_count=len(violations),
            )
            result = self._apply_all_constraints(result, constraints)

        # Final check after last iteration
        remaining = self.validate(result, constraints)
        if remaining:
            msg = (
                f"Unresolvable constraint violations after {self._max_repair_iterations} iterations: "
                + "; ".join(remaining[:5])
            )
            if self._raise_on_unresolvable:
                raise ValueError(msg)
            logger.warning("Unresolvable violations remain", violations=remaining[:5])

        return result

    async def apply_async(
        self,
        data: pd.DataFrame,
        constraints: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Apply and repair constraints asynchronously.

        Args:
            data: Synthetic DataFrame to transform.
            constraints: List of constraint descriptors.

        Returns:
            Transformed DataFrame with violations repaired.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self.apply, data, constraints),
        )

    def report_violations(
        self,
        data: pd.DataFrame,
        constraints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a structured violation report per constraint.

        Args:
            data: Synthetic DataFrame to inspect.
            constraints: List of constraint descriptors.

        Returns:
            Report dict with per-constraint pass/fail status and violation counts.
        """
        constraint_results: list[dict[str, Any]] = []
        total_violations = 0

        for constraint in constraints:
            constraint_type = constraint.get("constraint_type", "")
            column = constraint.get("column", "")
            params = constraint.get("params", {})

            try:
                violations = self._check_constraint(data, constraint_type, column, params)
                constraint_results.append(
                    {
                        "constraint_type": constraint_type,
                        "column": column,
                        "passed": len(violations) == 0,
                        "violation_count": len(violations),
                        "violations": violations[:10],
                    }
                )
                total_violations += len(violations)
            except Exception as exc:
                constraint_results.append(
                    {
                        "constraint_type": constraint_type,
                        "column": column,
                        "passed": False,
                        "violation_count": -1,
                        "error": str(exc),
                    }
                )
                total_violations += 1

        return {
            "total_constraints": len(constraints),
            "failed_constraints": sum(1 for r in constraint_results if not r["passed"]),
            "total_violations": total_violations,
            "all_passed": total_violations == 0,
            "results": constraint_results,
        }

    def enforce_foreign_key(
        self,
        data: pd.DataFrame,
        child_column: str,
        parent_values: list[Any],
    ) -> pd.DataFrame:
        """Replace FK values not in parent_values by sampling from valid set.

        Args:
            data: DataFrame with the child table column.
            child_column: Column containing foreign key values.
            parent_values: Valid parent primary key values to sample from.

        Returns:
            DataFrame with all FK values replaced with valid parent references.
        """
        if not parent_values or child_column not in data.columns:
            return data

        result = data.copy()
        invalid_mask = ~result[child_column].isin(parent_values)
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            replacement_values = np.random.choice(parent_values, size=invalid_count, replace=True)
            result.loc[invalid_mask, child_column] = replacement_values
            logger.info(
                "FK violations repaired",
                column=child_column,
                repaired_count=invalid_count,
            )

        return result

    def enforce_uniqueness(
        self,
        data: pd.DataFrame,
        column: str,
        existing_values: set[Any] | None = None,
    ) -> pd.DataFrame:
        """Deduplicate a column by dropping or regenerating duplicate rows.

        Duplicate rows beyond the first occurrence are dropped. If the column
        has overlapping values with existing_values (cross-batch uniqueness),
        those rows are also removed.

        Args:
            data: DataFrame to deduplicate.
            column: Column that must contain unique values.
            existing_values: Optional set of values already allocated externally.

        Returns:
            DataFrame with duplicates removed.
        """
        result = data.copy()

        if existing_values:
            overlap_mask = result[column].isin(existing_values)
            overlap_count = overlap_mask.sum()
            if overlap_count > 0:
                result = result[~overlap_mask].reset_index(drop=True)
                logger.info(
                    "Cross-batch uniqueness violations removed",
                    column=column,
                    removed_count=overlap_count,
                )

        before = len(result)
        result = result.drop_duplicates(subset=[column]).reset_index(drop=True)
        duplicates_removed = before - len(result)

        if duplicates_removed > 0:
            logger.info(
                "Duplicate rows removed for uniqueness",
                column=column,
                removed_count=duplicates_removed,
            )

        return result

    def enforce_inter_table_referential_integrity(
        self,
        tables: dict[str, pd.DataFrame],
        relationships: list[dict[str, str]],
    ) -> dict[str, pd.DataFrame]:
        """Enforce FK relationships across multiple synthesized tables.

        For each relationship, replaces invalid child FK values with randomly
        sampled parent PK values. Processes in dependency order (parents first).

        Args:
            tables: Dict mapping table_name → DataFrame.
            relationships: List of relationship dicts, each with:
                - parent_table: Name of the parent table.
                - parent_key: Parent primary key column.
                - child_table: Name of the child table.
                - child_key: Child foreign key column.

        Returns:
            Updated tables dict with all FK violations resolved.
        """
        result_tables = {name: df.copy() for name, df in tables.items()}

        for rel in relationships:
            parent_table = rel.get("parent_table", "")
            parent_key = rel.get("parent_key", "")
            child_table = rel.get("child_table", "")
            child_key = rel.get("child_key", "")

            if parent_table not in result_tables or child_table not in result_tables:
                logger.warning(
                    "Relationship table not found — skipping",
                    parent_table=parent_table,
                    child_table=child_table,
                )
                continue

            parent_values = result_tables[parent_table][parent_key].dropna().tolist()
            if not parent_values:
                logger.warning(
                    "Parent table has no values for FK repair",
                    parent_table=parent_table,
                    parent_key=parent_key,
                )
                continue

            result_tables[child_table] = self.enforce_foreign_key(
                data=result_tables[child_table],
                child_column=child_key,
                parent_values=parent_values,
            )

        return result_tables

    def _check_constraint(
        self,
        data: pd.DataFrame,
        constraint_type: str,
        column: str,
        params: dict[str, Any],
    ) -> list[str]:
        """Dispatch constraint validation to the appropriate handler.

        Args:
            data: DataFrame to check.
            constraint_type: Type identifier string.
            column: Target column name.
            params: Constraint parameters dict.

        Returns:
            List of violation messages.
        """
        if constraint_type == _CONSTRAINT_NOT_NULL:
            return self._check_not_null(data, column)
        elif constraint_type == _CONSTRAINT_RANGE:
            return self._check_range(data, column, params)
        elif constraint_type == _CONSTRAINT_ENUM:
            return self._check_enum(data, column, params)
        elif constraint_type == _CONSTRAINT_UNIQUE:
            return self._check_unique(data, column)
        elif constraint_type == _CONSTRAINT_FOREIGN_KEY:
            return self._check_foreign_key(data, column, params)
        elif constraint_type == _CONSTRAINT_REGEX:
            return self._check_regex(data, column, params)
        else:
            return [f"Unknown constraint_type '{constraint_type}' on column '{column}'"]

    def _check_not_null(self, data: pd.DataFrame, column: str) -> list[str]:
        """Check that a column has no null values.

        Args:
            data: DataFrame to check.
            column: Column name.

        Returns:
            Violation messages if nulls found.
        """
        if column not in data.columns:
            return [f"Column '{column}' not found in DataFrame"]
        null_count = data[column].isna().sum()
        if null_count > 0:
            return [f"Column '{column}' has {null_count} null values (not_null constraint)"]
        return []

    def _check_range(
        self,
        data: pd.DataFrame,
        column: str,
        params: dict[str, Any],
    ) -> list[str]:
        """Check that numeric values fall within [min, max].

        Args:
            data: DataFrame to check.
            column: Numeric column name.
            params: Dict with optional 'min' and 'max' keys.

        Returns:
            Violation messages if out-of-range values found.
        """
        if column not in data.columns:
            return [f"Column '{column}' not found in DataFrame"]

        violations: list[str] = []
        numeric_col = pd.to_numeric(data[column], errors="coerce").dropna()

        if "min" in params and (numeric_col < params["min"]).any():
            below_count = int((numeric_col < params["min"]).sum())
            violations.append(
                f"Column '{column}' has {below_count} values below min={params['min']}"
            )

        if "max" in params and (numeric_col > params["max"]).any():
            above_count = int((numeric_col > params["max"]).sum())
            violations.append(
                f"Column '{column}' has {above_count} values above max={params['max']}"
            )

        return violations

    def _check_enum(
        self,
        data: pd.DataFrame,
        column: str,
        params: dict[str, Any],
    ) -> list[str]:
        """Check that all values belong to an allowed set.

        Args:
            data: DataFrame to check.
            column: Categorical column name.
            params: Dict with 'allowed_values' key (list).

        Returns:
            Violation messages if disallowed values found.
        """
        if column not in data.columns:
            return [f"Column '{column}' not found in DataFrame"]

        allowed = set(params.get("allowed_values", []))
        if not allowed:
            return []

        invalid_mask = ~data[column].isin(allowed) & data[column].notna()
        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            invalid_sample = data.loc[invalid_mask, column].unique()[:5].tolist()
            return [
                f"Column '{column}' has {invalid_count} values not in allowed set. "
                f"Sample: {invalid_sample}"
            ]
        return []

    def _check_unique(self, data: pd.DataFrame, column: str) -> list[str]:
        """Check that all non-null values in a column are unique.

        Args:
            data: DataFrame to check.
            column: Column name.

        Returns:
            Violation messages if duplicates found.
        """
        if column not in data.columns:
            return [f"Column '{column}' not found in DataFrame"]

        non_null = data[column].dropna()
        duplicate_count = int(len(non_null) - non_null.nunique())

        if duplicate_count > 0:
            return [f"Column '{column}' has {duplicate_count} duplicate values (unique constraint)"]
        return []

    def _check_foreign_key(
        self,
        data: pd.DataFrame,
        column: str,
        params: dict[str, Any],
    ) -> list[str]:
        """Check that all FK values exist in the provided parent value set.

        Args:
            data: DataFrame to check.
            column: Child FK column name.
            params: Dict with 'parent_values' key (list of valid PKs).

        Returns:
            Violation messages if orphaned FK values found.
        """
        if column not in data.columns:
            return [f"Column '{column}' not found in DataFrame"]

        parent_values = set(params.get("parent_values", []))
        if not parent_values:
            return []

        invalid_mask = ~data[column].isin(parent_values) & data[column].notna()
        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            return [
                f"Column '{column}' has {invalid_count} FK values not in parent table "
                f"({len(parent_values)} valid parent keys)"
            ]
        return []

    def _check_regex(
        self,
        data: pd.DataFrame,
        column: str,
        params: dict[str, Any],
    ) -> list[str]:
        """Check that string values match a regex pattern.

        Args:
            data: DataFrame to check.
            column: String column name.
            params: Dict with 'pattern' key (regex string).

        Returns:
            Violation messages if pattern mismatches found.
        """
        import re

        if column not in data.columns:
            return [f"Column '{column}' not found in DataFrame"]

        pattern = params.get("pattern", "")
        if not pattern:
            return []

        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return [f"Invalid regex pattern for column '{column}': {exc}"]

        non_null = data[column].dropna().astype(str)
        mismatch_mask = ~non_null.str.match(compiled)
        mismatch_count = int(mismatch_mask.sum())

        if mismatch_count > 0:
            return [
                f"Column '{column}' has {mismatch_count} values not matching pattern '{pattern}'"
            ]
        return []

    def _apply_all_constraints(
        self,
        data: pd.DataFrame,
        constraints: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Apply all constraint repairs to the DataFrame in one pass.

        Args:
            data: DataFrame to repair.
            constraints: List of constraint descriptors.

        Returns:
            Repaired DataFrame.
        """
        result = data.copy()

        for constraint in constraints:
            constraint_type = constraint.get("constraint_type", "")
            column = constraint.get("column", "")
            params = constraint.get("params", {})

            if column not in result.columns:
                continue

            try:
                if constraint_type == _CONSTRAINT_RANGE:
                    result = self._repair_range(result, column, params)
                elif constraint_type == _CONSTRAINT_ENUM:
                    result = self._repair_enum(result, column, params)
                elif constraint_type == _CONSTRAINT_NOT_NULL:
                    result = self._repair_not_null(result, column, params)
                elif constraint_type == _CONSTRAINT_FOREIGN_KEY:
                    parent_values = params.get("parent_values", [])
                    result = self.enforce_foreign_key(result, column, parent_values)
                elif constraint_type == _CONSTRAINT_UNIQUE:
                    result = self.enforce_uniqueness(result, column)
            except Exception as exc:
                logger.error(
                    "Constraint repair failed",
                    constraint_type=constraint_type,
                    column=column,
                    error=str(exc),
                )

        return result

    def _repair_range(
        self,
        data: pd.DataFrame,
        column: str,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Clamp out-of-range numeric values to [min, max].

        Args:
            data: DataFrame to repair.
            column: Numeric column name.
            params: Dict with optional 'min' and 'max' keys.

        Returns:
            DataFrame with values clamped to valid range.
        """
        result = data.copy()
        numeric_col = pd.to_numeric(result[column], errors="coerce")

        if "min" in params:
            numeric_col = numeric_col.clip(lower=params["min"])
        if "max" in params:
            numeric_col = numeric_col.clip(upper=params["max"])

        result[column] = numeric_col
        return result

    def _repair_enum(
        self,
        data: pd.DataFrame,
        column: str,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Replace values not in allowed set by sampling from allowed values.

        Args:
            data: DataFrame to repair.
            column: Categorical column name.
            params: Dict with 'allowed_values' key.

        Returns:
            DataFrame with invalid values replaced by valid samples.
        """
        allowed = params.get("allowed_values", [])
        if not allowed:
            return data

        result = data.copy()
        invalid_mask = ~result[column].isin(allowed) & result[column].notna()
        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            replacement_values = np.random.choice(allowed, size=invalid_count, replace=True)
            result.loc[invalid_mask, column] = replacement_values

        return result

    def _repair_not_null(
        self,
        data: pd.DataFrame,
        column: str,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Replace null values with the column mode or a provided fill value.

        Args:
            data: DataFrame to repair.
            column: Column name.
            params: Dict with optional 'fill_value' key.

        Returns:
            DataFrame with null values filled.
        """
        result = data.copy()
        null_mask = result[column].isna()
        null_count = int(null_mask.sum())

        if null_count == 0:
            return result

        fill_value = params.get("fill_value")
        if fill_value is None:
            mode_series = result[column].mode()
            fill_value = mode_series.iloc[0] if len(mode_series) > 0 else 0

        result.loc[null_mask, column] = fill_value
        return result

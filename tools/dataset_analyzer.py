"""
DatasetAnalyzer tool.

Loads CSV files or built-in sklearn datasets and returns descriptive statistics,
data types, missing-value counts, and a sample of rows — all JSON-serialisable.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger("multiagent.tools.dataset_analyzer")

# Sklearn datasets available by name
_SKLEARN_LOADERS: Dict[str, str] = {
    "iris": "load_iris",
    "wine": "load_wine",
    "breast_cancer": "load_breast_cancer",
    "diabetes": "load_diabetes",
    "digits": "load_digits",
    "california_housing": "fetch_california_housing",
    "linnerud": "load_linnerud",
}


class DatasetAnalyzer:
    """Analyze CSV files or sklearn built-in datasets."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        source: str,
        dataset_name: Optional[str] = None,
        max_rows: int = 5,
    ) -> Dict[str, Any]:
        """
        Unified entry point.

        Parameters
        ----------
        source :
            One of:
            - ``"sklearn:<dataset_name>"``  e.g. ``"sklearn:iris"``
            - ``"csv:<filepath>"``          e.g. ``"csv:/data/sales.csv"``
            - A bare file path (CSV assumed)
        dataset_name :
            Explicit sklearn dataset name; overrides ``source`` prefix parsing.
        max_rows :
            Number of sample rows to return.
        """
        if dataset_name:
            return self._analyze_sklearn(dataset_name, max_rows)
        if source.startswith("sklearn:"):
            return self._analyze_sklearn(source[8:].strip(), max_rows)
        if source.startswith("csv:"):
            return self._analyze_csv(source[4:].strip(), max_rows)
        # Default: treat as CSV filepath
        return self._analyze_csv(source.strip(), max_rows)

    def list_sklearn_datasets(self) -> List[str]:
        return list(_SKLEARN_LOADERS.keys())

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def _analyze_csv(self, path: str, max_rows: int) -> Dict[str, Any]:
        try:
            import pandas as pd
        except ImportError:
            return {"success": False, "error": "pandas is not installed"}

        try:
            df = pd.read_csv(path)
            return self._summarize(df, source=f"csv:{path}", max_rows=max_rows)
        except FileNotFoundError:
            return {"success": False, "error": f"File not found: {path}"}
        except Exception as exc:          # noqa: BLE001
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Sklearn
    # ------------------------------------------------------------------

    def _analyze_sklearn(self, name: str, max_rows: int) -> Dict[str, Any]:
        try:
            import pandas as pd
            from sklearn import datasets as sk_datasets
        except ImportError as exc:
            return {"success": False, "error": f"Missing dependency: {exc}"}

        name_lower = name.lower()
        if name_lower not in _SKLEARN_LOADERS:
            return {
                "success": False,
                "error": (
                    f"Unknown sklearn dataset '{name}'. "
                    f"Available: {list(_SKLEARN_LOADERS)}"
                ),
            }

        loader_name = _SKLEARN_LOADERS[name_lower]
        try:
            loader = getattr(sk_datasets, loader_name)
            bunch = loader()
            df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
            if hasattr(bunch, "target"):
                df["target"] = bunch.target
            if hasattr(bunch, "target_names"):
                target_names = list(bunch.target_names)
            else:
                target_names = None
            result = self._summarize(df, source=f"sklearn:{name}", max_rows=max_rows)
            if target_names:
                result["target_names"] = target_names
            return result
        except Exception as exc:          # noqa: BLE001
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Core summary builder
    # ------------------------------------------------------------------

    def _summarize(self, df: Any, source: str, max_rows: int) -> Dict[str, Any]:
        import pandas as pd

        # Descriptive statistics — convert to JSON-safe dict
        desc = df.describe(include="all")
        desc_clean: Dict[str, Dict[str, Any]] = {}
        for col in desc.columns:
            desc_clean[str(col)] = {
                str(idx): _safe_scalar(v)
                for idx, v in desc[col].items()
            }

        # Correlation matrix (numeric only)
        corr: Dict[str, Dict[str, Any]] = {}
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) > 1:
            corr_df = df[num_cols].corr()
            for col in corr_df.columns:
                corr[str(col)] = {
                    str(idx): _safe_scalar(v)
                    for idx, v in corr_df[col].items()
                }

        return {
            "success": True,
            "source": source,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
            "missing_values": {
                col: int(cnt) for col, cnt in df.isnull().sum().items()
            },
            "descriptive_stats": desc_clean,
            "correlation_matrix": corr,
            "sample_rows": _df_to_records(df.head(max_rows)),
            "memory_usage_kb": round(
                df.memory_usage(deep=True).sum() / 1024, 2
            ),
        }

    # ------------------------------------------------------------------
    # Auto-Experimentation
    # ------------------------------------------------------------------

    def auto_experiment(
        self,
        source: str,
        target_column: str = "target",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Automatically train and compare multiple ML classifiers on a dataset.

        Tries:
          - RandomForestClassifier
          - LogisticRegression
          - GradientBoostingClassifier
          - XGBClassifier (if xgboost is installed)

        Returns a ranked leaderboard and selects the best model by accuracy.

        Parameters
        ----------
        source        : Same format as ``analyze()`` — "sklearn:<name>" or "csv:<path>"
        target_column : Column to use as the label (default: "target")
        test_size     : Fraction held out for testing (default: 0.2)
        random_state  : Seed for reproducibility
        """
        try:
            import pandas as pd
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
        except ImportError as exc:
            return {"success": False, "error": f"Missing dependency: {exc}"}

        # ── Load data ──────────────────────────────────────────────────
        if source.startswith("sklearn:"):
            result = self._analyze_sklearn(source[8:].strip(), max_rows=0)
        elif source.startswith("csv:"):
            result = self._analyze_csv(source[4:].strip(), max_rows=0)
        else:
            result = self._analyze_csv(source.strip(), max_rows=0)

        if not result.get("success"):
            return result

        # Reload as DataFrame (the analyze methods summarise; we need raw data)
        try:
            if source.startswith("sklearn:"):
                from sklearn import datasets as sk_datasets
                name = source[8:].strip().lower()
                loader = getattr(sk_datasets, _SKLEARN_LOADERS[name])
                bunch = loader()
                df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
                df[target_column] = bunch.target
            else:
                path = source[4:].strip() if source.startswith("csv:") else source
                df = pd.read_csv(path)
        except Exception as exc:           # noqa: BLE001
            return {"success": False, "error": f"Failed to load data: {exc}"}

        if target_column not in df.columns:
            return {
                "success": False,
                "error": (
                    f"Target column '{target_column}' not found. "
                    f"Available columns: {list(df.columns)}"
                ),
            }

        # ── Prepare features / labels ──────────────────────────────────
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode string labels
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # ── Model roster ───────────────────────────────────────────────
        candidates: Dict[str, Any] = {
            "RandomForest":       RandomForestClassifier(
                n_estimators=100, random_state=random_state
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=random_state
            ),
            "GradientBoosting":   GradientBoostingClassifier(
                n_estimators=100, random_state=random_state
            ),
        }

        # Optional: XGBoost
        try:
            from xgboost import XGBClassifier
            candidates["XGBoost"] = XGBClassifier(
                n_estimators=100, random_state=random_state,
                eval_metric="logloss", verbosity=0,
            )
        except ImportError:
            pass

        # ── Train + evaluate each model ────────────────────────────────
        leaderboard: List[Dict[str, Any]] = []
        errors: Dict[str, str] = {}

        for name, model in candidates.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc  = round(float(accuracy_score(y_test, preds)), 4)
                f1   = round(float(f1_score(
                    y_test, preds, average="weighted", zero_division=0
                )), 4)
                leaderboard.append({
                    "model":    name,
                    "accuracy": acc,
                    "f1_weighted": f1,
                })
                logger.info(
                    "[DatasetAnalyzer] auto_experiment | %s → acc=%.4f f1=%.4f",
                    name, acc, f1,
                )
            except Exception as exc:       # noqa: BLE001
                errors[name] = str(exc)
                logger.warning(
                    "[DatasetAnalyzer] auto_experiment | %s failed: %s", name, exc
                )

        if not leaderboard:
            return {
                "success": False,
                "error": "All models failed.",
                "model_errors": errors,
            }

        # ── Select best by accuracy ────────────────────────────────────
        leaderboard.sort(key=lambda x: x["accuracy"], reverse=True)
        best = leaderboard[0]

        summary_lines = [
            f"Auto-Experiment Results — {source}",
            f"Dataset: {result['shape'][0]} rows × {len(X.columns)} features",
            f"Train/test split: {int((1-test_size)*100)}% / {int(test_size*100)}%",
            "",
            "Leaderboard:",
        ]
        for rank, entry in enumerate(leaderboard, 1):
            summary_lines.append(
                f"  {rank}. {entry['model']:<22} "
                f"accuracy={entry['accuracy']:.4f}  "
                f"f1={entry['f1_weighted']:.4f}"
            )
        summary_lines += [
            "",
            f"Best model: {best['model']}",
            f"  Accuracy : {best['accuracy']:.4f}",
            f"  F1 (wtd) : {best['f1_weighted']:.4f}",
        ]
        if errors:
            summary_lines.append(f"Skipped (errors): {list(errors.keys())}")

        return {
            "success":    True,
            "source":     source,
            "leaderboard": leaderboard,
            "best_model": best,
            "summary":    "\n".join(summary_lines),
            "model_errors": errors,
        }

    # ------------------------------------------------------------------
    # Formatting helper (for agent prompts)
    # ------------------------------------------------------------------

    def format_result(self, result: Dict[str, Any]) -> str:
        if not result.get("success"):
            return f"DatasetAnalyzer ERROR: {result.get('error', 'unknown')}"
        lines = [
            f"Dataset: {result['source']}",
            f"Shape: {result['shape'][0]} rows × {result['shape'][1]} columns",
            f"Columns: {', '.join(result['columns'])}",
            "Missing values: "
            + ", ".join(
                f"{c}={v}" for c, v in result["missing_values"].items() if v > 0
            )
            or "None",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_scalar(v: Any) -> Any:
    """Convert NaN / ±inf to None for JSON safety."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _df_to_records(df: Any) -> List[Dict[str, Any]]:
    """Convert DataFrame head to a JSON-safe list of dicts."""
    records = []
    for row in df.to_dict(orient="records"):
        records.append({str(k): _safe_scalar(v) for k, v in row.items()})
    return records


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[DatasetAnalyzer] = None


def get_dataset_analyzer() -> DatasetAnalyzer:
    global _instance
    if _instance is None:
        _instance = DatasetAnalyzer()
    return _instance

"""
Unit tests for DatasetAnalyzer — sklearn source, CSV source, auto_experiment.
No mocking of sklearn/pandas — uses real lightweight datasets.
"""
import os
import tempfile

import pandas as pd
import pytest

from tools.dataset_analyzer import DatasetAnalyzer, _safe_scalar, _df_to_records


@pytest.fixture
def analyzer():
    return DatasetAnalyzer()


@pytest.fixture
def tmp_csv(tmp_path):
    df = pd.DataFrame({
        "age":    [25, 30, 35, 40, 45],
        "salary": [50000, 60000, 70000, 80000, 90000],
        "target": [0, 1, 0, 1, 0],
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# list_sklearn_datasets
# ---------------------------------------------------------------------------

class TestListSklearnDatasets:
    def test_returns_list(self, analyzer):
        result = analyzer.list_sklearn_datasets()
        assert isinstance(result, list)

    def test_includes_iris(self, analyzer):
        assert "iris" in analyzer.list_sklearn_datasets()

    def test_includes_wine(self, analyzer):
        assert "wine" in analyzer.list_sklearn_datasets()


# ---------------------------------------------------------------------------
# analyze — sklearn source
# ---------------------------------------------------------------------------

class TestAnalyzeSklearn:
    def test_success_flag_true(self, analyzer):
        result = analyzer.analyze("sklearn:iris")
        assert result["success"] is True

    def test_shape_is_list(self, analyzer):
        result = analyzer.analyze("sklearn:iris")
        assert isinstance(result["shape"], list)
        assert len(result["shape"]) == 2

    def test_iris_row_count(self, analyzer):
        result = analyzer.analyze("sklearn:iris")
        assert result["shape"][0] == 150

    def test_columns_is_list(self, analyzer):
        result = analyzer.analyze("sklearn:iris")
        assert isinstance(result["columns"], list)

    def test_target_column_present(self, analyzer):
        result = analyzer.analyze("sklearn:iris")
        assert "target" in result["columns"]

    def test_target_names_present(self, analyzer):
        result = analyzer.analyze("sklearn:iris")
        assert "target_names" in result

    def test_missing_values_dict(self, analyzer):
        result = analyzer.analyze("sklearn:iris")
        assert isinstance(result["missing_values"], dict)

    def test_descriptive_stats_present(self, analyzer):
        result = analyzer.analyze("sklearn:iris")
        assert "descriptive_stats" in result

    def test_sample_rows_length(self, analyzer):
        result = analyzer.analyze("sklearn:iris", max_rows=3)
        assert len(result["sample_rows"]) == 3

    def test_dataset_name_kwarg(self, analyzer):
        result = analyzer.analyze("", dataset_name="wine")
        assert result["success"] is True

    def test_unknown_dataset_returns_failure(self, analyzer):
        result = analyzer.analyze("sklearn:doesnotexist")
        assert result["success"] is False
        assert "Unknown sklearn dataset" in result["error"]

    def test_source_field_matches(self, analyzer):
        result = analyzer.analyze("sklearn:iris")
        assert result["source"] == "sklearn:iris"

    def test_memory_usage_positive(self, analyzer):
        result = analyzer.analyze("sklearn:iris")
        assert result["memory_usage_kb"] > 0

    def test_correlation_matrix_present(self, analyzer):
        result = analyzer.analyze("sklearn:iris")
        assert isinstance(result["correlation_matrix"], dict)


# ---------------------------------------------------------------------------
# analyze — CSV source
# ---------------------------------------------------------------------------

class TestAnalyzeCsv:
    def test_success_flag(self, analyzer, tmp_csv):
        result = analyzer.analyze(f"csv:{tmp_csv}")
        assert result["success"] is True

    def test_shape_correct(self, analyzer, tmp_csv):
        result = analyzer.analyze(f"csv:{tmp_csv}")
        assert result["shape"] == [5, 3]

    def test_columns_correct(self, analyzer, tmp_csv):
        result = analyzer.analyze(f"csv:{tmp_csv}")
        assert set(result["columns"]) == {"age", "salary", "target"}

    def test_no_missing_values(self, analyzer, tmp_csv):
        result = analyzer.analyze(f"csv:{tmp_csv}")
        assert all(v == 0 for v in result["missing_values"].values())

    def test_bare_path_treated_as_csv(self, analyzer, tmp_csv):
        result = analyzer.analyze(tmp_csv)
        assert result["success"] is True

    def test_file_not_found_returns_failure(self, analyzer):
        result = analyzer.analyze("csv:/nonexistent/path/data.csv")
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# auto_experiment
# ---------------------------------------------------------------------------

class TestAutoExperiment:
    def test_success_on_iris(self, analyzer):
        result = analyzer.auto_experiment("sklearn:iris")
        assert result["success"] is True

    def test_leaderboard_is_list(self, analyzer):
        result = analyzer.auto_experiment("sklearn:iris")
        assert isinstance(result["leaderboard"], list)

    def test_leaderboard_has_entries(self, analyzer):
        result = analyzer.auto_experiment("sklearn:iris")
        assert len(result["leaderboard"]) >= 1

    def test_leaderboard_sorted_by_accuracy(self, analyzer):
        result = analyzer.auto_experiment("sklearn:iris")
        scores = [e["accuracy"] for e in result["leaderboard"]]
        assert scores == sorted(scores, reverse=True)

    def test_best_model_key_present(self, analyzer):
        result = analyzer.auto_experiment("sklearn:iris")
        assert "best_model" in result
        assert "model" in result["best_model"]

    def test_best_model_accuracy_in_range(self, analyzer):
        result = analyzer.auto_experiment("sklearn:iris")
        acc = result["best_model"]["accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_summary_string_present(self, analyzer):
        result = analyzer.auto_experiment("sklearn:iris")
        assert isinstance(result["summary"], str)
        assert "Best model" in result["summary"]

    def test_auto_experiment_csv(self, analyzer, tmp_csv):
        result = analyzer.auto_experiment(f"csv:{tmp_csv}", target_column="target")
        assert result["success"] is True

    def test_missing_target_column(self, analyzer, tmp_csv):
        result = analyzer.auto_experiment(f"csv:{tmp_csv}", target_column="nonexistent")
        assert result["success"] is False
        assert "Target column" in result["error"]

    def test_invalid_source_returns_failure(self, analyzer):
        result = analyzer.auto_experiment("sklearn:fakeds")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_safe_scalar_nan_returns_none(self):
        import math
        assert _safe_scalar(float("nan")) is None

    def test_safe_scalar_inf_returns_none(self):
        assert _safe_scalar(float("inf")) is None

    def test_safe_scalar_normal_float(self):
        assert _safe_scalar(3.14) == 3.14

    def test_safe_scalar_int(self):
        assert _safe_scalar(42) == 42

    def test_df_to_records_length(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        records = _df_to_records(df)
        assert len(records) == 2

    def test_df_to_records_keys(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        assert set(_df_to_records(df)[0].keys()) == {"x", "y"}

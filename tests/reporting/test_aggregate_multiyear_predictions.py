"""Tests for vercye_ops.reporting.aggregate_multiyear_predictions"""

import os

import pandas as pd
import pytest

from vercye_ops.reporting.aggregate_multiyear_predictions import (
    aggregate_years,
    collect_files,
    get_avaiable_agg_levels,
    get_available_timepoints,
    merge_preds_gt_yearly,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


DEFAULT_STUDY_ID = "teststudy"


def _create_dir_structure(base, years_timepoints, agg_levels=None, study_id=DEFAULT_STUDY_ID):
    """Create a realistic directory structure with prediction and GT files.

    Prediction filename schema (matches the pipeline):
        agg_yield_estimates_{level_name}_{study_id}_{year}_{timepoint}.csv

    years_timepoints: dict like {"2022": ["T-0", "T-1"], "2023": ["T-0"]}
    agg_levels: list like ["county", "state"], if None defaults to ["county"]
    """
    if agg_levels is None:
        agg_levels = ["county"]

    for year, timepoints in years_timepoints.items():
        for tp in timepoints:
            tp_dir = os.path.join(base, year, tp)
            os.makedirs(tp_dir, exist_ok=True)
            for agg in agg_levels:
                pred_df = pd.DataFrame(
                    {
                        "region": ["A", "B"],
                        "mean_yield_kg_ha": [1000, 2000],
                    }
                )
                pred_df.to_csv(
                    os.path.join(
                        tp_dir,
                        f"agg_yield_estimates_{agg}_{study_id}_{year}_{tp}.csv",
                    ),
                    index=False,
                )

        # GT file per year per agg level
        for agg in agg_levels:
            gt_df = pd.DataFrame(
                {
                    "region": ["A", "B"],
                    "reported_mean_yield_kg_ha": [1100, 1900],
                }
            )
            gt_df.to_csv(
                os.path.join(base, year, f"referencedata_{agg}-{year}.csv"),
                index=False,
            )


# ---------------------------------------------------------------------------
# get_available_timepoints
# ---------------------------------------------------------------------------


class TestGetAvailableTimepoints:
    def test_finds_timepoints(self, tmp_path):
        _create_dir_structure(str(tmp_path), {"2022": ["T-0", "T-1"], "2023": ["T-0"]})
        tps = get_available_timepoints(str(tmp_path))
        assert set(tps) == {"T-0", "T-1"}

    def test_empty_dir(self, tmp_path):
        tps = get_available_timepoints(str(tmp_path))
        assert tps == []

    def test_ignores_files_at_year_level(self, tmp_path):
        _create_dir_structure(str(tmp_path), {"2022": ["T-0"]})
        # Add a file (not dir) at year level
        (tmp_path / "2022" / "some_file.txt").write_text("hi")
        tps = get_available_timepoints(str(tmp_path))
        assert "some_file.txt" not in tps


# ---------------------------------------------------------------------------
# get_avaiable_agg_levels
# ---------------------------------------------------------------------------


class TestGetAvailableAggLevels:
    def test_finds_agg_levels(self, tmp_path):
        _create_dir_structure(str(tmp_path), {"2022": ["T-0"]}, agg_levels=["county", "state"])
        levels = get_avaiable_agg_levels(str(tmp_path))
        assert set(levels) == {"county", "state"}

    def test_deduplicates(self, tmp_path):
        _create_dir_structure(str(tmp_path), {"2022": ["T-0", "T-1"]}, agg_levels=["county"])
        levels = get_avaiable_agg_levels(str(tmp_path))
        assert levels == ["county"]


# ---------------------------------------------------------------------------
# collect_files
# ---------------------------------------------------------------------------


class TestCollectFiles:
    def test_collects_pred_and_gt(self, tmp_path):
        _create_dir_structure(str(tmp_path), {"2022": ["T-0"]})
        pred_paths, gt_paths = collect_files(str(tmp_path), "county", "T-0")
        assert "2022" in pred_paths
        assert "2022" in gt_paths

    def test_missing_gt_still_returns_preds(self, tmp_path):
        _create_dir_structure(str(tmp_path), {"2022": ["T-0"]})
        # Remove GT file
        os.remove(os.path.join(str(tmp_path), "2022", "referencedata_county-2022.csv"))
        pred_paths, gt_paths = collect_files(str(tmp_path), "county", "T-0")
        assert "2022" in pred_paths
        assert "2022" not in gt_paths

    def test_raises_on_duplicate_pred_files(self, tmp_path):
        _create_dir_structure(str(tmp_path), {"2022": ["T-0"]})
        # Create a second file that also matches the collect_files glob for
        # county/2022/T-0 but with a different study_id.
        tp_dir = os.path.join(str(tmp_path), "2022", "T-0")
        pd.DataFrame({"region": ["A"]}).to_csv(
            os.path.join(tp_dir, "agg_yield_estimates_county_otherstudy_2022_T-0.csv"),
            index=False,
        )
        with pytest.raises(Exception, match="More than one"):
            collect_files(str(tmp_path), "county", "T-0")


# ---------------------------------------------------------------------------
# merge_preds_gt_yearly
# ---------------------------------------------------------------------------


class TestMergePredsGtYearly:
    def test_merge_adds_year_column(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        gt_path = tmp_path / "gt.csv"
        pd.DataFrame({"region": ["A"], "mean_yield_kg_ha": [1000]}).to_csv(pred_path, index=False)
        pd.DataFrame({"region": ["A"], "reported_mean_yield_kg_ha": [1100]}).to_csv(gt_path, index=False)

        dfs = merge_preds_gt_yearly(
            {"2022": str(pred_path)},
            {"2022": str(gt_path)},
        )
        assert len(dfs) == 1
        assert "year" in dfs[0].columns
        assert dfs[0]["year"].iloc[0] == "2022"

    def test_merge_without_gt(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        pd.DataFrame({"region": ["A"], "mean_yield_kg_ha": [1000]}).to_csv(pred_path, index=False)

        dfs = merge_preds_gt_yearly({"2022": str(pred_path)}, {})
        assert len(dfs) == 1
        assert "reported_mean_yield_kg_ha" not in dfs[0].columns

    def test_existing_year_column_renamed(self, tmp_path):
        """If the prediction CSV already has a 'year' column, it should be
        preserved as 'year_original' and the new year column added."""
        pred_path = tmp_path / "pred.csv"
        pd.DataFrame({"region": ["A"], "mean_yield_kg_ha": [1000], "year": [2021]}).to_csv(pred_path, index=False)
        gt_path = tmp_path / "gt.csv"
        pd.DataFrame({"region": ["A"], "reported_mean_yield_kg_ha": [1100]}).to_csv(gt_path, index=False)

        dfs = merge_preds_gt_yearly({"2022": str(pred_path)}, {"2022": str(gt_path)})
        assert "year_original" in dfs[0].columns
        assert dfs[0]["year"].iloc[0] == "2022"


# ---------------------------------------------------------------------------
# aggregate_years (integration)
# ---------------------------------------------------------------------------


class TestAggregateYears:
    def test_multi_year_aggregation(self, tmp_path):
        _create_dir_structure(str(tmp_path), {"2022": ["T-0"], "2023": ["T-0"]})
        result = aggregate_years(str(tmp_path), "county", "T-0")
        assert len(result) == 4  # 2 regions x 2 years
        assert set(result["year"]) == {"2022", "2023"}

    def test_no_preds_returns_empty(self, tmp_path):
        os.makedirs(str(tmp_path / "2022" / "T-0"))
        result = aggregate_years(str(tmp_path), "county", "T-0")
        assert result.empty

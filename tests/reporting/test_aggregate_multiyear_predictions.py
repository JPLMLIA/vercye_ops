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
        levels = get_avaiable_agg_levels(str(tmp_path), DEFAULT_STUDY_ID)
        assert set(levels) == {"county", "state"}

    def test_deduplicates(self, tmp_path):
        _create_dir_structure(str(tmp_path), {"2022": ["T-0", "T-1"]}, agg_levels=["county"])
        levels = get_avaiable_agg_levels(str(tmp_path), DEFAULT_STUDY_ID)
        assert levels == ["county"]


# ---------------------------------------------------------------------------
# collect_files
# ---------------------------------------------------------------------------


class TestCollectFiles:
    def test_collects_pred_and_gt(self, tmp_path):
        _create_dir_structure(str(tmp_path), {"2022": ["T-0"]})
        pred_paths, gt_paths = collect_files(str(tmp_path), "county", "T-0", DEFAULT_STUDY_ID)
        assert "2022" in pred_paths
        assert "2022" in gt_paths

    def test_missing_gt_still_returns_preds(self, tmp_path):
        _create_dir_structure(str(tmp_path), {"2022": ["T-0"]})
        # Remove GT file
        os.remove(os.path.join(str(tmp_path), "2022", "referencedata_county-2022.csv"))
        pred_paths, gt_paths = collect_files(str(tmp_path), "county", "T-0", DEFAULT_STUDY_ID)
        assert "2022" in pred_paths
        assert "2022" not in gt_paths

    def test_other_studys_file_is_ignored(self, tmp_path):
        """A file for the same level but a different study_id must not be picked up.

        This used to raise "More than one ..." because the level was matched with a
        wildcard that also absorbed the study_id; the level name is now matched
        exactly, so another study's file is simply not ours.
        """
        _create_dir_structure(str(tmp_path), {"2022": ["T-0"]})
        tp_dir = os.path.join(str(tmp_path), "2022", "T-0")
        pd.DataFrame({"region": ["A"]}).to_csv(
            os.path.join(tp_dir, "agg_yield_estimates_county_otherstudy_2022_T-0.csv"),
            index=False,
        )
        pred_paths, _ = collect_files(str(tmp_path), "county", "T-0", DEFAULT_STUDY_ID)
        assert list(pred_paths) == ["2022"]
        assert DEFAULT_STUDY_ID in os.path.basename(pred_paths["2022"])

    def test_level_that_is_a_prefix_of_another_is_not_confused(self, tmp_path):
        """"county" must not swallow "county_subset" (the ADM1 / ADM1_ThreeCounties bug)."""
        _create_dir_structure(str(tmp_path), {"2022": ["T-0"]}, agg_levels=["county", "county_subset"])
        for lvl in ("county", "county_subset"):
            pred_paths, _ = collect_files(str(tmp_path), lvl, "T-0", DEFAULT_STUDY_ID)
            assert len(pred_paths) == 1
            assert os.path.basename(pred_paths["2022"]).startswith(
                f"agg_yield_estimates_{lvl}_{DEFAULT_STUDY_ID}_"
            )


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

    def test_duplicate_gt_region_raises(self, tmp_path):
        """A reference file with a duplicated region key must not silently fan the
        prediction out into a many-to-many cartesian product; the merge should fail.
        """
        pred_path = tmp_path / "pred.csv"
        gt_path = tmp_path / "gt.csv"
        pd.DataFrame({"region": ["A", "B"], "mean_yield_kg_ha": [1000, 2000]}).to_csv(pred_path, index=False)
        # Region A duplicated with two different reported values.
        pd.DataFrame(
            {"region": ["A", "A", "B"], "reported_mean_yield_kg_ha": [1100, 1200, 1900]}
        ).to_csv(gt_path, index=False)

        with pytest.raises(Exception):
            merge_preds_gt_yearly({"2022": str(pred_path)}, {"2022": str(gt_path)})

    def test_reference_already_in_pred_no_duplicate_columns(self, tmp_path):
        """When the per-year prediction file already carries reported_mean_yield_kg_ha
        (the pipeline merges reference upstream), re-merging the reference must not
        create _x/_y duplicate columns nor multiply rows.
        """
        pred_path = tmp_path / "pred.csv"
        gt_path = tmp_path / "gt.csv"
        pd.DataFrame(
            {"region": ["A", "B"], "mean_yield_kg_ha": [1000, 2000], "reported_mean_yield_kg_ha": [1100, 1900]}
        ).to_csv(pred_path, index=False)
        pd.DataFrame({"region": ["A", "B"], "reported_mean_yield_kg_ha": [1100, 1900]}).to_csv(gt_path, index=False)

        dfs = merge_preds_gt_yearly({"2022": str(pred_path)}, {"2022": str(gt_path)})
        df = dfs[0]
        assert len(df) == 2
        assert "reported_mean_yield_kg_ha_x" not in df.columns
        assert "reported_mean_yield_kg_ha_y" not in df.columns
        assert "reported_mean_yield_kg_ha" in df.columns
        assert sorted(df["reported_mean_yield_kg_ha"].tolist()) == [1100, 1900]

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
        result = aggregate_years(str(tmp_path), "county", "T-0", DEFAULT_STUDY_ID)
        assert len(result) == 4  # 2 regions x 2 years
        assert set(result["year"]) == {"2022", "2023"}

    def test_no_preds_returns_empty(self, tmp_path):
        os.makedirs(str(tmp_path / "2022" / "T-0"))
        result = aggregate_years(str(tmp_path), "county", "T-0", DEFAULT_STUDY_ID)
        assert result.empty

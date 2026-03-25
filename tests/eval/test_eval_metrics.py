"""Tests for vercye_ops.evaluation.evaluate_yield_estimates"""

import numpy as np
import pandas as pd
import pytest

from vercye_ops.evaluation.evaluate_yield_estimates import (
    compute_errors_per_region,
    compute_metrics,
    get_preds_obs,
    write_errors,
    write_metrics,
)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions(self):
        """When preds == obs, errors should be zero and R2 should be 1."""
        obs = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        preds = obs.copy()
        m = compute_metrics(preds, obs)

        assert m["n_regions"] == 5
        assert m["mean_err_kg_ha"] == 0.0
        assert m["median_err_kg_ha"] == 0.0
        assert m["mean_abs_err_kg_ha"] == 0.0
        assert m["rmse_kg_ha"] == 0.0
        assert m["mape"] == 0.0
        assert m["r2_scikit"] == 1.0
        assert m["r2_rsq_excel"] == 1.0

    def test_known_values(self):
        """Verify metrics against hand-calculated values."""
        obs = np.array([100.0, 200.0, 300.0])
        preds = np.array([110.0, 190.0, 310.0])

        m = compute_metrics(preds, obs)

        # errors: [10, -10, 10]
        assert m["mean_err_kg_ha"] == pytest.approx(10 / 3 * 3 / 3, abs=0.5)  # ~3.3
        assert m["n_regions"] == 3

        # RMSE = sqrt(mean([100, 100, 100])) = 10
        assert m["rmse_kg_ha"] == pytest.approx(10.0, abs=0.1)

        # RRMSE = 10 / 200 * 100 = 5.0%
        assert m["rrmse"] == pytest.approx(5.0, abs=0.1)

    def test_constant_bias(self):
        """A constant positive bias should show in mean error."""
        obs = np.array([1000.0, 2000.0, 3000.0])
        preds = obs + 100.0
        m = compute_metrics(preds, obs)

        assert m["mean_err_kg_ha"] == pytest.approx(100.0, abs=0.1)
        assert m["rmse_kg_ha"] == pytest.approx(100.0, abs=0.1)
        # R2 (scikit) should still be high since pattern is captured
        assert m["r2_rsq_excel"] == pytest.approx(1.0, abs=0.01)

    def test_empty_arrays_return_none_metrics(self):
        m = compute_metrics(np.array([]), np.array([])  )
        assert m["n_regions"] == 0
        assert all(v is None for k, v in m.items() if k != "n_regions")

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="Length"):
            compute_metrics(np.array([1, 2]), np.array([1]))

    def test_single_element(self):
        """Single element: R2 is undefined / degenerate but should not crash."""
        m = compute_metrics(np.array([100.0]), np.array([120.0]))
        assert m["n_regions"] == 1
        assert m["rmse_kg_ha"] == pytest.approx(20.0, abs=0.1)

    def test_negative_r2_scikit(self):
        """When predictions are worse than predicting the mean, r2_scikit < 0."""
        obs = np.array([100.0, 200.0, 300.0, 400.0])
        preds = np.array([400.0, 300.0, 200.0, 100.0])  # reversed
        m = compute_metrics(preds, obs)
        assert m["r2_scikit"] < 0

    def test_all_metrics_keys_present(self):
        obs = np.array([100.0, 200.0])
        preds = np.array([110.0, 210.0])
        m = compute_metrics(preds, obs)
        expected_keys = {
            "n_regions", "mape", "mean_err_kg_ha", "median_err_kg_ha",
            "mean_abs_err_kg_ha", "median_abs_err_kg_ha", "rmse_kg_ha",
            "rrmse", "r2_scikit", "r2_rsq_excel", "r2_scikit_bestfit",
        }
        assert set(m.keys()) == expected_keys

    def test_r2_rsq_excel_vs_r2_scikit_differ(self):
        """r2_rsq_excel (Pearson^2) and r2_scikit (sklearn) can differ when
        predictions have a systematic bias."""
        obs = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        preds = obs * 0.5 + 500  # linear but biased
        m = compute_metrics(preds, obs)
        # Pearson R^2 should be 1.0 (perfect linear correlation)
        assert m["r2_rsq_excel"] == pytest.approx(1.0, abs=0.01)
        # sklearn R2 penalizes bias, so it will be lower
        assert m["r2_scikit"] < m["r2_rsq_excel"]


# ---------------------------------------------------------------------------
# compute_errors_per_region
# ---------------------------------------------------------------------------


class TestComputeErrorsPerRegion:
    def test_basic_errors(self):
        preds = np.array([110.0, 190.0])
        obs = np.array([100.0, 200.0])
        regions = np.array(["A", "B"])
        result = compute_errors_per_region(preds, obs, regions)

        np.testing.assert_allclose(result["error_kg_ha"], [10.0, -10.0], atol=0.1)
        np.testing.assert_allclose(result["rel_error_percent"], [10.0, -5.0], atol=0.1)

    def test_empty_input(self):
        result = compute_errors_per_region(np.array([]), np.array([]), np.array([]))
        assert result["error_kg_ha"] is None
        assert result["rel_error_percent"] is None

    def test_zero_observation_division(self):
        """Division by zero when obs=0 should produce inf, not crash."""
        preds = np.array([100.0])
        obs = np.array([0.0])
        regions = np.array(["A"])
        result = compute_errors_per_region(preds, obs, regions)
        assert np.isinf(result["rel_error_percent"][0])


# ---------------------------------------------------------------------------
# write_metrics / write_errors
# ---------------------------------------------------------------------------


class TestWriteMetricsAndErrors:
    def test_write_metrics_roundtrip(self, tmp_path):
        metrics = {"rmse_kg_ha": 123.4, "r2_scikit": 0.95, "n_regions": 5}
        fpath = tmp_path / "metrics.csv"
        write_metrics(metrics, str(fpath))
        df = pd.read_csv(fpath)
        assert df["rmse_kg_ha"].iloc[0] == pytest.approx(123.4)

    def test_write_errors_roundtrip(self, tmp_path):
        errors = {
            "error_kg_ha": np.array([10.0, -10.0]),
            "rel_error_percent": np.array([5.0, -2.5]),
            "region": ["A", "B"],
        }
        fpath = tmp_path / "errors.csv"
        write_errors(errors, str(fpath))
        df = pd.read_csv(fpath)
        assert len(df) == 2
        assert list(df["region"]) == ["A", "B"]


# ---------------------------------------------------------------------------
# get_preds_obs
# ---------------------------------------------------------------------------


class TestGetPredsObs:
    def _make_csvs(self, tmp_path, gt_data, est_data):
        gt_path = tmp_path / "gt.csv"
        est_path = tmp_path / "est.csv"
        pd.DataFrame(gt_data).to_csv(gt_path, index=False)
        pd.DataFrame(est_data).to_csv(est_path, index=False)
        return str(gt_path), str(est_path)

    def test_basic_merge(self, tmp_path):
        gt = {"region": ["A", "B"], "reported_mean_yield_kg_ha": [1000, 2000]}
        est = {"region": ["A", "B"], "mean_yield_kg_ha": [1100, 1900], "total_area_ha": [10, 20]}
        gt_path, est_path = self._make_csvs(tmp_path, gt, est)

        result = get_preds_obs(est_path, gt_path, pixel_converted=True)
        assert len(result["preds"]) == 2
        assert len(result["obs"]) == 2

    def test_pixel_converted_false_uses_apsim_column(self, tmp_path):
        gt = {"region": ["A"], "reported_mean_yield_kg_ha": [1000]}
        est = {
            "region": ["A"],
            "mean_yield_kg_ha": [1100],
            "apsim_mean_yield_estimate_kg_ha": [900],
            "total_area_ha": [10],
        }
        gt_path, est_path = self._make_csvs(tmp_path, gt, est)

        result = get_preds_obs(est_path, gt_path, pixel_converted=False)
        assert result["preds"].iloc[0] == 900

    def test_computes_reported_yield_from_production(self, tmp_path):
        """When reported_mean_yield_kg_ha is absent, it should be derived from
        reported_production_kg / total_area_ha (from the merged dataframe)."""
        gt = {"region": ["A"], "reported_production_kg": [50000]}
        est = {"region": ["A"], "mean_yield_kg_ha": [5100], "total_area_ha": [10]}
        gt_path, est_path = self._make_csvs(tmp_path, gt, est)

        result = get_preds_obs(est_path, gt_path, pixel_converted=True)
        assert result["obs"].iloc[0] == pytest.approx(5000.0)

    def test_nan_obs_dropped(self, tmp_path):
        gt = {"region": ["A", "B"], "reported_mean_yield_kg_ha": [1000, float("nan")]}
        est = {"region": ["A", "B"], "mean_yield_kg_ha": [1100, 1900], "total_area_ha": [10, 20]}
        gt_path, est_path = self._make_csvs(tmp_path, gt, est)

        result = get_preds_obs(est_path, gt_path, pixel_converted=True)
        assert len(result["preds"]) == 1

    def test_missing_predictions_column_raises(self, tmp_path):
        gt = {"region": ["A"], "reported_mean_yield_kg_ha": [1000]}
        est = {"region": ["A"], "wrong_col": [1100]}
        gt_path, est_path = self._make_csvs(tmp_path, gt, est)

        with pytest.raises(ValueError, match="mean_yield_kg_ha"):
            get_preds_obs(est_path, gt_path, pixel_converted=True)

    def test_no_reported_yield_and_no_production_raises(self, tmp_path):
        gt = {"region": ["A"], "some_other_col": [42]}
        est = {"region": ["A"], "mean_yield_kg_ha": [1100], "total_area_ha": [10]}
        gt_path, est_path = self._make_csvs(tmp_path, gt, est)

        with pytest.raises(ValueError, match="reported_production_kg"):
            get_preds_obs(est_path, gt_path, pixel_converted=True)

    def test_regions_not_in_both_files_are_dropped(self, tmp_path):
        """Inner join should drop regions only in one file."""
        gt = {"region": ["A", "B", "C"], "reported_mean_yield_kg_ha": [1000, 2000, 3000]}
        est = {"region": ["A", "C"], "mean_yield_kg_ha": [1100, 3100], "total_area_ha": [10, 30]}
        gt_path, est_path = self._make_csvs(tmp_path, gt, est)

        result = get_preds_obs(est_path, gt_path, pixel_converted=True)
        assert len(result["preds"]) == 2
        assert set(result["region"]) == {"A", "C"}

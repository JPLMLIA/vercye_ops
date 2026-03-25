"""Tests for vercye_ops.reporting.aggregate_yield_estimates_per_eval_lvl"""

import pandas as pd
import pytest

from vercye_ops.reporting.aggregate_yield_estimates_per_eval_lvl import aggregate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_estimation_df(data_dicts):
    """Build a DataFrame with the columns that aggregate() expects."""
    return pd.DataFrame(data_dicts)


# ---------------------------------------------------------------------------
# Tests for weighted_mean
# ---------------------------------------------------------------------------


class TestAggregateWeightedMean:
    def test_uniform_weights(self):
        """With equal areas, weighted mean == simple mean."""
        df = _make_estimation_df([
            {"county": "A", "mean_yield_kg_ha": 1000, "total_area_ha": 10,
             "total_yield_production_kg": 10000, "total_yield_production_ton": 10.0,
             "apsim_mean_yield_estimate_kg_ha": 900},
            {"county": "A", "mean_yield_kg_ha": 2000, "total_area_ha": 10,
             "total_yield_production_kg": 20000, "total_yield_production_ton": 20.0,
             "apsim_mean_yield_estimate_kg_ha": 1800},
        ])
        result = aggregate(df, "county")
        assert result["mean_yield_kg_ha"].iloc[0] == pytest.approx(1500.0)

    def test_non_uniform_weights(self):
        """Weighted mean should skew toward the higher-area region."""
        df = _make_estimation_df([
            {"county": "A", "mean_yield_kg_ha": 1000, "total_area_ha": 90,
             "total_yield_production_kg": 90000, "total_yield_production_ton": 90.0,
             "apsim_mean_yield_estimate_kg_ha": 900},
            {"county": "A", "mean_yield_kg_ha": 2000, "total_area_ha": 10,
             "total_yield_production_kg": 20000, "total_yield_production_ton": 20.0,
             "apsim_mean_yield_estimate_kg_ha": 1800},
        ])
        result = aggregate(df, "county")
        # (1000*90 + 2000*10) / 100 = 1100
        assert result["mean_yield_kg_ha"].iloc[0] == pytest.approx(1100.0)

    def test_total_area_is_summed(self):
        df = _make_estimation_df([
            {"county": "A", "mean_yield_kg_ha": 1000, "total_area_ha": 50,
             "total_yield_production_kg": 50000, "total_yield_production_ton": 50.0,
             "apsim_mean_yield_estimate_kg_ha": 900},
            {"county": "A", "mean_yield_kg_ha": 2000, "total_area_ha": 30,
             "total_yield_production_kg": 60000, "total_yield_production_ton": 60.0,
             "apsim_mean_yield_estimate_kg_ha": 1800},
        ])
        result = aggregate(df, "county")
        assert result["total_area_ha"].iloc[0] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Tests for weighted_median
# ---------------------------------------------------------------------------


class TestAggregateWeightedMedian:
    def test_weighted_median_two_equal_weights(self):
        """With two equal-weight entries, median should be one of them
        (the first where cumulative weight >= half)."""
        df = _make_estimation_df([
            {"county": "A", "mean_yield_kg_ha": 1000, "total_area_ha": 50,
             "total_yield_production_kg": 50000, "total_yield_production_ton": 50.0,
             "apsim_mean_yield_estimate_kg_ha": 900},
            {"county": "A", "mean_yield_kg_ha": 3000, "total_area_ha": 50,
             "total_yield_production_kg": 150000, "total_yield_production_ton": 150.0,
             "apsim_mean_yield_estimate_kg_ha": 2700},
        ])
        result = aggregate(df, "county")
        # Sorted values: [1000, 3000], cumulative weights: [50, 100], half=50
        # First index where cumsum >= 50 is the first entry = 1000
        assert result["median_yield_kg_ha"].iloc[0] == pytest.approx(1000.0)

    def test_weighted_median_skewed(self):
        """Heavier weight on high-yield region should shift median up."""
        df = _make_estimation_df([
            {"county": "A", "mean_yield_kg_ha": 1000, "total_area_ha": 10,
             "total_yield_production_kg": 10000, "total_yield_production_ton": 10.0,
             "apsim_mean_yield_estimate_kg_ha": 900},
            {"county": "A", "mean_yield_kg_ha": 3000, "total_area_ha": 90,
             "total_yield_production_kg": 270000, "total_yield_production_ton": 270.0,
             "apsim_mean_yield_estimate_kg_ha": 2700},
        ])
        result = aggregate(df, "county")
        # Sorted: [1000(w=10), 3000(w=90)], cumsum=[10, 100], half=50
        # 3000 is where cumsum >= 50
        assert result["median_yield_kg_ha"].iloc[0] == pytest.approx(3000.0)

    def test_weighted_median_three_entries(self):
        df = _make_estimation_df([
            {"county": "A", "mean_yield_kg_ha": 1000, "total_area_ha": 30,
             "total_yield_production_kg": 30000, "total_yield_production_ton": 30.0,
             "apsim_mean_yield_estimate_kg_ha": 900},
            {"county": "A", "mean_yield_kg_ha": 2000, "total_area_ha": 40,
             "total_yield_production_kg": 80000, "total_yield_production_ton": 80.0,
             "apsim_mean_yield_estimate_kg_ha": 1800},
            {"county": "A", "mean_yield_kg_ha": 5000, "total_area_ha": 30,
             "total_yield_production_kg": 150000, "total_yield_production_ton": 150.0,
             "apsim_mean_yield_estimate_kg_ha": 4500},
        ])
        result = aggregate(df, "county")
        # Sorted: [1000(30), 2000(40), 5000(30)], cumsum=[30, 70, 100], half=50
        # 2000 is where cumsum >= 50
        assert result["median_yield_kg_ha"].iloc[0] == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# Tests for multiple groups
# ---------------------------------------------------------------------------


class TestAggregateMultipleGroups:
    def test_two_groups(self):
        df = _make_estimation_df([
            {"county": "X", "mean_yield_kg_ha": 1000, "total_area_ha": 10,
             "total_yield_production_kg": 10000, "total_yield_production_ton": 10.0,
             "apsim_mean_yield_estimate_kg_ha": 900},
            {"county": "Y", "mean_yield_kg_ha": 2000, "total_area_ha": 20,
             "total_yield_production_kg": 40000, "total_yield_production_ton": 40.0,
             "apsim_mean_yield_estimate_kg_ha": 1800},
        ])
        result = aggregate(df, "county")
        assert len(result) == 2
        assert set(result["region"]) == {"X", "Y"}

    def test_column_renamed_to_region(self):
        df = _make_estimation_df([
            {"state": "CA", "mean_yield_kg_ha": 1000, "total_area_ha": 10,
             "total_yield_production_kg": 10000, "total_yield_production_ton": 10.0,
             "apsim_mean_yield_estimate_kg_ha": 900},
        ])
        result = aggregate(df, "state")
        assert "region" in result.columns
        assert "state" not in result.columns

    def test_production_summed_per_group(self):
        df = _make_estimation_df([
            {"county": "A", "mean_yield_kg_ha": 1000, "total_area_ha": 10,
             "total_yield_production_kg": 10000, "total_yield_production_ton": 10.0,
             "apsim_mean_yield_estimate_kg_ha": 900},
            {"county": "A", "mean_yield_kg_ha": 2000, "total_area_ha": 20,
             "total_yield_production_kg": 40000, "total_yield_production_ton": 40.0,
             "apsim_mean_yield_estimate_kg_ha": 1800},
        ])
        result = aggregate(df, "county")
        assert result["total_yield_production_kg"].iloc[0] == 50000
        assert result["total_yield_production_ton"].iloc[0] == pytest.approx(50.0, abs=0.01)


# ---------------------------------------------------------------------------
# End-to-end weighted aggregation correctness
# Tests that the weighting chain is scientifically sound: pixel masking
# determines area, area determines weight, weight determines aggregated yield.
# ---------------------------------------------------------------------------


class TestWeightedAggregationEndToEnd:
    """Verify that aggregation weights are correct in realistic scenarios.

    Context: Each region's mean_yield_kg_ha comes from nanmean over only
    masked-in pixels, and total_area_ha counts only those pixels.  When
    regions are aggregated (e.g. to state level), total_area_ha is used
    as the weight.  A region with more cropland pixels should have more
    influence on the aggregated yield.
    """

    def test_large_region_dominates_weighted_mean(self):
        """A region with 10x the area should dominate the weighted mean."""
        df = _make_estimation_df([
            # Small region: 10 ha, high yield
            {"state": "TX", "mean_yield_kg_ha": 5000, "total_area_ha": 10,
             "total_yield_production_kg": 50000, "total_yield_production_ton": 50.0,
             "apsim_mean_yield_estimate_kg_ha": 4500},
            # Large region: 100 ha, low yield
            {"state": "TX", "mean_yield_kg_ha": 1000, "total_area_ha": 100,
             "total_yield_production_kg": 100000, "total_yield_production_ton": 100.0,
             "apsim_mean_yield_estimate_kg_ha": 900},
        ])
        result = aggregate(df, "state")
        wm = result["mean_yield_kg_ha"].iloc[0]
        # Weighted mean = (5000*10 + 1000*100) / 110 = 150000/110 ~ 1363.6
        assert wm == pytest.approx(150000 / 110, abs=0.1)
        # Must be much closer to 1000 than to 5000
        assert abs(wm - 1000) < abs(wm - 5000)

    def test_unweighted_mean_would_give_wrong_answer(self):
        """Show that a simple (unweighted) mean gives a different—incorrect—answer."""
        df = _make_estimation_df([
            {"state": "TX", "mean_yield_kg_ha": 5000, "total_area_ha": 1,
             "total_yield_production_kg": 5000, "total_yield_production_ton": 5.0,
             "apsim_mean_yield_estimate_kg_ha": 4500},
            {"state": "TX", "mean_yield_kg_ha": 1000, "total_area_ha": 99,
             "total_yield_production_kg": 99000, "total_yield_production_ton": 99.0,
             "apsim_mean_yield_estimate_kg_ha": 900},
        ])
        result = aggregate(df, "state")
        weighted = result["mean_yield_kg_ha"].iloc[0]
        unweighted = (5000 + 1000) / 2  # 3000 - would be scientifically wrong

        # Weighted should be ~1040, far from unweighted 3000
        assert weighted == pytest.approx((5000 * 1 + 1000 * 99) / 100, abs=0.1)
        assert abs(weighted - unweighted) > 1000  # proves unweighted is wrong

    def test_single_region_weighted_mean_equals_input(self):
        """A single sub-region's weighted mean should equal its own mean yield."""
        df = _make_estimation_df([
            {"state": "NY", "mean_yield_kg_ha": 3456, "total_area_ha": 42,
             "total_yield_production_kg": 145152, "total_yield_production_ton": 145.152,
             "apsim_mean_yield_estimate_kg_ha": 3100},
        ])
        result = aggregate(df, "state")
        assert result["mean_yield_kg_ha"].iloc[0] == pytest.approx(3456.0)
        assert result["median_yield_kg_ha"].iloc[0] == pytest.approx(3456.0)

    def test_total_production_independent_of_weighting(self):
        """Total production (kg and ton) should just be summed, not weighted."""
        df = _make_estimation_df([
            {"state": "CA", "mean_yield_kg_ha": 2000, "total_area_ha": 50,
             "total_yield_production_kg": 100000, "total_yield_production_ton": 100.0,
             "apsim_mean_yield_estimate_kg_ha": 1800},
            {"state": "CA", "mean_yield_kg_ha": 4000, "total_area_ha": 50,
             "total_yield_production_kg": 200000, "total_yield_production_ton": 200.0,
             "apsim_mean_yield_estimate_kg_ha": 3600},
        ])
        result = aggregate(df, "state")
        # Production is additive, not weighted
        assert result["total_yield_production_kg"].iloc[0] == 300000
        assert result["total_yield_production_ton"].iloc[0] == pytest.approx(300.0)
        # But mean yield IS weighted (both equal area so equal to simple mean here)
        assert result["mean_yield_kg_ha"].iloc[0] == pytest.approx(3000.0)

    def test_apsim_estimate_also_area_weighted(self):
        """apsim_mean_yield_estimate_kg_ha should use the same area weighting."""
        df = _make_estimation_df([
            {"state": "KS", "mean_yield_kg_ha": 2000, "total_area_ha": 80,
             "total_yield_production_kg": 160000, "total_yield_production_ton": 160.0,
             "apsim_mean_yield_estimate_kg_ha": 1800},
            {"state": "KS", "mean_yield_kg_ha": 4000, "total_area_ha": 20,
             "total_yield_production_kg": 80000, "total_yield_production_ton": 80.0,
             "apsim_mean_yield_estimate_kg_ha": 3600},
        ])
        result = aggregate(df, "state")
        expected_apsim = (1800 * 80 + 3600 * 20) / 100
        assert result["apsim_mean_yield_estimate_kg_ha"].iloc[0] == pytest.approx(expected_apsim)

    def test_negative_area_corrupts_weighting(self):
        """If a region has area=-1 (the all-NaN sentinel), it would corrupt
        the weighted mean. This documents the problem with using -1 as a
        sentinel for total_area_ha.

        A -1 area means: (a) negative weight in the weighted mean, and
        (b) the total area sum is wrong.
        """
        df = _make_estimation_df([
            {"state": "TX", "mean_yield_kg_ha": 0, "total_area_ha": -1,
             "total_yield_production_kg": 0, "total_yield_production_ton": 0.0,
             "apsim_mean_yield_estimate_kg_ha": 0},
            {"state": "TX", "mean_yield_kg_ha": 3000, "total_area_ha": 100,
             "total_yield_production_kg": 300000, "total_yield_production_ton": 300.0,
             "apsim_mean_yield_estimate_kg_ha": 2700},
        ])
        result = aggregate(df, "state")
        # With area=-1, weighted mean = (0*-1 + 3000*100) / 99 = 3030.3
        # instead of the correct 3000.0
        wm = result["mean_yield_kg_ha"].iloc[0]
        assert wm != pytest.approx(3000.0, abs=1.0), (
            "If this passes, the -1 sentinel is no longer causing issues"
        )
        # Total area is also wrong: 99 instead of 100
        assert result["total_area_ha"].iloc[0] == pytest.approx(99.0)

"""Tests for vercye_ops.reporting.aggregate_meteorological_stats

NOTE: This file includes a test that demonstrates a BUG in read_met_file():
The missing-data check uses the wrong column index for each variable.
See test_missing_data_marker_bug for details.
"""

import pandas as pd
import pytest

from vercye_ops.reporting.aggregate_meteorological_stats import (
    aggregate_data,
    read_met_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MET_HEADER = """\
[weather.met.weather]
latitude = 48.50
longitude = 30.00

[weather.met.weather]
year day radn meant maxt mint rain wind
() () (MJ/m^2) (oC) (oC) (oC) (mm) (m/s)
"""


def _write_met_file(path, data_lines, header=None):
    if header is None:
        header = MET_HEADER
    with open(str(path), "w") as f:
        f.write(header)
        for line in data_lines:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# read_met_file
# ---------------------------------------------------------------------------


class TestReadMetFile:
    def test_basic_parsing(self, tmp_path):
        fpath = tmp_path / "test.met"
        _write_met_file(fpath, [
            "2022  1  15.0  5.0  10.0  0.0  2.5  3.0",
            "2022  2  16.0  6.0  11.0  1.0  0.0  2.5",
        ])
        df = read_met_file(str(fpath))

        assert len(df) == 2
        assert df["latitude"].iloc[0] == pytest.approx(48.50)
        assert df["longitude"].iloc[0] == pytest.approx(30.00)
        assert df["radn"].iloc[0] == pytest.approx(15.0)
        assert df["meant"].iloc[0] == pytest.approx(5.0)
        assert df["maxt"].iloc[0] == pytest.approx(10.0)
        assert df["mint"].iloc[0] == pytest.approx(0.0)
        assert df["rain"].iloc[0] == pytest.approx(2.5)
        assert df["wind"].iloc[0] == pytest.approx(3.0)

    def test_date_parsing(self, tmp_path):
        fpath = tmp_path / "test.met"
        _write_met_file(fpath, [
            "2022  32  15.0  5.0  10.0  0.0  2.5  3.0",  # Feb 1
        ])
        df = read_met_file(str(fpath))
        assert df["date"].iloc[0].month == 2
        assert df["date"].iloc[0].day == 1

    def test_missing_data_marker_bug(self, tmp_path):
        """BUG: read_met_file checks the WRONG column index for missing data.

        Line 47: maxt uses `if float(parts[3]) != -999` (checks meant, not maxt)
        Line 48: mint uses `if float(parts[4]) != -999` (checks maxt, not mint)
        Line 49: rain uses `if float(parts[5]) != -999` (checks mint, not rain)
        Line 50: wind uses `if float(parts[6]) != -999` (checks rain, not wind)

        This means if maxt is -999 but meant is valid, maxt will be read as -999
        instead of being replaced with 0.

        Example: meant=5.0 but maxt=-999 -> maxt should be 0, but because the
        check looks at parts[3] (meant=5.0 which != -999), maxt is set to -999.
        """
        fpath = tmp_path / "test.met"
        # maxt is -999 (missing), but meant (parts[3]) is valid
        _write_met_file(fpath, [
            "2022  1  15.0  5.0  -999  0.0  2.5  3.0",
        ])
        df = read_met_file(str(fpath))

        # Due to the bug: maxt will be -999 instead of 0, because the code
        # checks parts[3] (meant=5.0) instead of parts[4] (maxt=-999)
        # If the bug is fixed, this assertion should change to == 0.0
        assert df["maxt"].iloc[0] == pytest.approx(-999.0), (
            "This test documents the off-by-one bug in read_met_file. "
            "maxt=-999 is not caught because the code checks parts[3] (meant) instead of parts[4] (maxt)."
        )

    def test_missing_data_marker_radn_works_correctly(self, tmp_path):
        """radn is the only variable where the missing-data check is correct
        (it checks its own column parts[2])."""
        fpath = tmp_path / "test.met"
        _write_met_file(fpath, [
            "2022  1  -999  5.0  10.0  0.0  2.5  3.0",
        ])
        df = read_met_file(str(fpath))
        assert df["radn"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# aggregate_data
# ---------------------------------------------------------------------------


class TestAggregateData:
    def test_single_location_single_year(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-01", "2022-01-02"]),
            "latitude": [48.5, 48.5],
            "longitude": [30.0, 30.0],
            "radn": [15.0, 17.0],
            "meant": [5.0, 7.0],
            "maxt": [10.0, 12.0],
            "mint": [0.0, 2.0],
            "rain": [2.0, 4.0],
            "wind": [3.0, 5.0],
        })
        result = aggregate_data(df)

        assert len(result) == 1
        assert result["radn"].iloc[0] == pytest.approx(16.0)
        assert result["meant"].iloc[0] == pytest.approx(6.0)
        assert result["maxt"].iloc[0] == pytest.approx(11.0)
        # Wind should be converted from m/s to km/h (*3.6)
        assert result["wind"].iloc[0] == pytest.approx(4.0 * 3.6)

    def test_rainfall_is_annual_average(self):
        """Rainfall should be summed per year, then averaged across years."""
        dates = pd.to_datetime([
            "2021-06-01", "2021-06-02",  # year 1: total rain = 10
            "2022-06-01", "2022-06-02",  # year 2: total rain = 20
        ])
        df = pd.DataFrame({
            "date": dates,
            "latitude": [48.5] * 4,
            "longitude": [30.0] * 4,
            "radn": [15.0] * 4,
            "meant": [5.0] * 4,
            "maxt": [10.0] * 4,
            "mint": [0.0] * 4,
            "rain": [5.0, 5.0, 10.0, 10.0],
            "wind": [3.0] * 4,
        })
        result = aggregate_data(df)
        # Annual rain: 2021=10, 2022=20, average=15
        assert result["rain"].iloc[0] == pytest.approx(15.0)

    def test_multiple_locations(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-01", "2022-01-01"]),
            "latitude": [48.5, 49.0],
            "longitude": [30.0, 31.0],
            "radn": [15.0, 20.0],
            "meant": [5.0, 10.0],
            "maxt": [10.0, 15.0],
            "mint": [0.0, 5.0],
            "rain": [2.0, 8.0],
            "wind": [3.0, 6.0],
        })
        result = aggregate_data(df)
        assert len(result) == 2

    def test_wind_conversion_m_s_to_km_h(self):
        """Verify wind is converted: 1 m/s = 3.6 km/h."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-01"]),
            "latitude": [48.5],
            "longitude": [30.0],
            "radn": [15.0],
            "meant": [5.0],
            "maxt": [10.0],
            "mint": [0.0],
            "rain": [2.0],
            "wind": [10.0],
        })
        result = aggregate_data(df)
        assert result["wind"].iloc[0] == pytest.approx(36.0)

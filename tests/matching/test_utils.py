"""Tests for vercye_ops.matching_sim_real.utils"""

import sqlite3

import pandas as pd
import pytest

from vercye_ops.matching_sim_real.utils import (
    build_default_query,
    compute_pixel_area,
    load_simulation_data,
)


# ---------------------------------------------------------------------------
# build_default_query
# ---------------------------------------------------------------------------


class TestBuildDefaultQuery:
    def test_wheat(self):
        q = build_default_query("Wheat")
        assert "Wheat.Leaf.LAI" in q
        assert "Yield" in q
        assert "SimulationID" in q

    def test_maize(self):
        q = build_default_query("Maize")
        assert "Maize.Leaf.LAI" in q


# ---------------------------------------------------------------------------
# compute_pixel_area
# ---------------------------------------------------------------------------


class TestComputePixelArea:
    def test_known_pixel_area_at_equator(self):
        """At the equator, 1 degree ~ 111 km, so a 0.01-degree pixel ~ 1.11 km.
        Area should be approximately 1.11 * 1.11 = ~1.23 km^2 = 1_230_000 m^2."""
        area = compute_pixel_area(
            lon=0.0, lat=0.0,
            pixel_width_deg=0.01, pixel_height_deg=0.01,
            output_crs="EPSG:32631",  # UTM zone 31N (near equator)
        )
        # ~1.23 km^2 = 1,230,000 m^2 with some tolerance
        assert 1_100_000 < area < 1_400_000

    def test_pixel_area_shrinks_at_high_latitude(self):
        """Pixel area in degrees should be smaller (in m^2) at higher latitudes
        due to meridian convergence."""
        area_equator = compute_pixel_area(
            lon=0.0, lat=0.0, pixel_width_deg=0.01, pixel_height_deg=0.01,
            output_crs="EPSG:32631",
        )
        area_60n = compute_pixel_area(
            lon=10.0, lat=60.0, pixel_width_deg=0.01, pixel_height_deg=0.01,
            output_crs="EPSG:32632",
        )
        assert area_60n < area_equator

    def test_zero_pixel_size(self):
        area = compute_pixel_area(
            lon=10.0, lat=45.0, pixel_width_deg=0.0, pixel_height_deg=0.0,
            output_crs="EPSG:32632",
        )
        assert area == 0.0

    def test_non_default_input_crs(self):
        """Should work with a non-WGS84 input CRS (though unusual)."""
        # Just verifying it doesn't crash
        area = compute_pixel_area(
            lon=500000.0, lat=5000000.0,
            pixel_width_deg=100, pixel_height_deg=100,
            output_crs="EPSG:32632",
            input_crs="EPSG:32632",
        )
        assert area == pytest.approx(10000.0)  # 100 * 100 in same CRS


# ---------------------------------------------------------------------------
# load_simulation_data
# ---------------------------------------------------------------------------


def _create_test_db(db_path, rows, extra_columns=None):
    """Create a minimal APSIM-like SQLite DB for testing."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cols = "SimulationID INTEGER, `Clock.Today` TEXT, `Clock.Today.DayOfYear` INTEGER, `Wheat.Leaf.LAI` REAL, Yield REAL"
    if extra_columns:
        for col_name, col_type in extra_columns:
            cols += f", `{col_name}` {col_type}"

    cursor.execute(f"CREATE TABLE Report ({cols})")

    for row in rows:
        placeholders = ", ".join(["?"] * len(row))
        cursor.execute(f"INSERT INTO Report VALUES ({placeholders})", row)

    conn.commit()
    conn.close()


class TestLoadSimulationData:
    def test_basic_load(self, tmp_path):
        db = tmp_path / "test.db"
        _create_test_db(db, [
            (1, "2023-01-15", 15, 0.5, 3000),
            (1, "2023-02-15", 46, 1.2, 3000),
            (2, "2023-01-15", 15, 0.3, 2500),
        ])
        df = load_simulation_data(str(db), crop_name="wheat")
        assert len(df) == 3
        assert "Wheat.Leaf.LAI" in df.columns

    def test_requires_crop_or_query(self):
        with pytest.raises(ValueError, match="Either"):
            load_simulation_data("dummy.db")

    def test_exact_duplicates_dropped(self, tmp_path):
        db = tmp_path / "test.db"
        _create_test_db(db, [
            (1, "2023-01-15", 15, 0.5, 3000),
            (1, "2023-01-15", 15, 0.5, 3000),  # exact duplicate
        ])
        df = load_simulation_data(str(db), crop_name="wheat")
        assert len(df) == 1

    def test_resolvable_duplicates_with_wheat_total_wt(self, tmp_path):
        """Duplicates that differ only in Wheat.Total.Wt should be resolved
        by keeping the row with the highest Wheat.Total.Wt."""
        db = tmp_path / "test.db"
        _create_test_db(
            db,
            [
                (1, "2023-01-15", 15, 0.5, 3000, 100.0),
                (1, "2023-01-15", 15, 0.5, 3000, 200.0),  # higher Wt
            ],
            extra_columns=[("Wheat.Total.Wt", "REAL")],
        )
        df = load_simulation_data(
            str(db),
            query="SELECT SimulationID, `Clock.Today`, `Clock.Today.DayOfYear`, `Wheat.Leaf.LAI`, Yield, `Wheat.Total.Wt` FROM Report",
        )
        assert len(df) == 1
        assert df["Wheat.Total.Wt"].iloc[0] == 200.0

    def test_unresolvable_duplicates_raise(self, tmp_path):
        """Duplicates differing in non-allowed columns should raise."""
        db = tmp_path / "test.db"
        _create_test_db(db, [
            (1, "2023-01-15", 15, 0.5, 3000),
            (1, "2023-01-15", 15, 0.5, 4000),  # different Yield
        ])
        with pytest.raises(ValueError, match="Duplicate"):
            load_simulation_data(str(db), crop_name="wheat")

    def test_date_index_is_set(self, tmp_path):
        db = tmp_path / "test.db"
        _create_test_db(db, [
            (1, "2023-01-15", 15, 0.5, 3000),
        ])
        df = load_simulation_data(str(db), crop_name="wheat")
        assert df.index.name == "Date"
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_custom_query(self, tmp_path):
        db = tmp_path / "test.db"
        _create_test_db(db, [
            (1, "2023-01-15", 15, 0.5, 3000),
        ])
        df = load_simulation_data(
            str(db),
            query="SELECT SimulationID, `Clock.Today`, Yield FROM Report",
        )
        assert "Yield" in df.columns
        assert len(df) == 1

    def test_crop_name_case_insensitive(self, tmp_path):
        db = tmp_path / "test.db"
        _create_test_db(db, [
            (1, "2023-01-15", 15, 0.5, 3000),
        ])
        df = load_simulation_data(str(db), crop_name="WHEAT")
        assert len(df) == 1

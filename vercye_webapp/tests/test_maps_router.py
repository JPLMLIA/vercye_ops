"""Tests for the results-map data endpoints.

The point of the new map is that it is assembled from small pieces instead of one
self-contained HTML - so the thing most worth guarding is that splitting it up did not
change any number. Every assertion about values compares the API's output against the
same ``agg_yield_estimates_*.csv`` the PDF reports and the legacy map read.

A synthetic study is built on a tmp_path so the suite does not depend on a real run
being present on the box.
"""

import json
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

WEBAPP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEBAPP))

STUDY = "teststudy"
YEAR, PREV_YEAR, TP, LEVEL = "2024", "2023", "T-0", "ADM1"
REGIONS = {"Alpha": (0.0, 0.0), "Beta": (1.0, 0.0), "Gamma": (2.0, 0.0)}
YIELDS = {"Alpha": 1500.0, "Beta": 2500.0, "Gamma": 900.0}


@pytest.fixture(scope="module")
def study_root(tmp_path_factory):
    """A study laid out the way the pipeline leaves one."""
    base = tmp_path_factory.mktemp("studies")
    root = base / STUDY / STUDY
    ytp = root / YEAR / TP
    ytp.mkdir(parents=True)
    (root / "aggregation_shapefiles").mkdir()

    # Aggregation shapefile: one row per region PER YEAR, as the real ones are.
    rows = []
    for name, (x, y) in REGIONS.items():
        for yr in (2023, 2024):
            rows.append({"shapeName": name, "year": yr, "geometry": box(x, y, x + 0.9, y + 0.9)})
    gpd.GeoDataFrame(rows, crs="EPSG:4326").to_file(
        root / "aggregation_shapefiles" / "levels.geojson", driver="GeoJSON"
    )

    # Primary-level boundaries, written per year/timepoint.
    gpd.GeoDataFrame(
        [{"region": n, "geometry": box(x, y, x + 0.9, y + 0.9)} for n, (x, y) in REGIONS.items()],
        crs="EPSG:4326",
    ).to_file(ytp / f"aggregated_region_boundaries_{STUDY}_{YEAR}_{TP}.geojson", driver="GeoJSON")

    # Stats. A level whose name is a prefix of another is included on purpose.
    for lvl in (LEVEL, f"{LEVEL}_Extra"):
        rows = [
            {
                "region": n,
                "mean_yield_kg_ha": YIELDS[n] if lvl == LEVEL else 1.0,
                "total_production_kg": YIELDS[n] * 10,
                # Partially missing, as a real column with no reference data would be.
                "reported_mean_yield_kg_ha": YIELDS[n] * 0.9 if n != "Gamma" else np.nan,
            }
            for n in REGIONS
        ]
        pd.DataFrame(rows).to_csv(ytp / f"agg_yield_estimates_{lvl}_{STUDY}_{YEAR}_{TP}.csv", index=False)

    # The primary-level file lists every region of the *source* shapefile, so regions the
    # study never simulated appear as all-NaN rows. Real studies have hundreds of these.
    prim = [{"region": n, "mean_yield_kg_ha": YIELDS[n], "total_production_kg": YIELDS[n] * 10} for n in REGIONS]
    prim += [{"region": f"unsimulated_{i}", "mean_yield_kg_ha": np.nan, "total_production_kg": np.nan} for i in range(4)]
    pd.DataFrame(prim).to_csv(ytp / f"agg_yield_estimates_primary_{STUDY}_{YEAR}_{TP}.csv", index=False)

    # A small COG-ish raster covering the regions.
    data = np.linspace(0, 3000, 256 * 256, dtype="float32").reshape(1, 256, 256)
    with rasterio.open(
        ytp / f"yield_mosaic_4326_{STUDY}_{YEAR}_{TP}.tif",
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(0, 0, 3, 1, 256, 256),
        nodata=0,
    ) as dst:
        dst.write(data)

    # Region plot zip.
    with zipfile.ZipFile(ytp / f"interactive_map_{STUDY}_{YEAR}_{TP}.zip", "w") as z:
        for n in REGIONS:
            z.writestr(f"{n}_yield_report.png", b"\x89PNG\r\n\x1a\n" + n.encode())

    # A prior year, so the study spans more than one and the multiyear view has a series.
    prev = root / PREV_YEAR / TP
    prev.mkdir(parents=True)
    pd.DataFrame(
        [{"region": n, "mean_yield_kg_ha": YIELDS[n] * 0.8, "reported_mean_yield_kg_ha": YIELDS[n] * 0.7}
         for n in REGIONS]
    ).to_csv(prev / f"agg_yield_estimates_{LEVEL}_{STUDY}_{PREV_YEAR}_{TP}.csv", index=False)
    gpd.GeoDataFrame(
        [{"region": n, "geometry": box(x, y, x + 0.9, y + 0.9)} for n, (x, y) in REGIONS.items()],
        crs="EPSG:4326",
    ).to_file(prev / f"aggregated_region_boundaries_{STUDY}_{PREV_YEAR}_{TP}.geojson", driver="GeoJSON")

    # The pipeline's own accuracy metrics, one row per level/year. The API reports these
    # verbatim - nothing in the map recomputes them.
    for yr, r2 in ((YEAR, 0.812), (PREV_YEAR, 0.604)):
        pd.DataFrame(
            [{
                "n_regions": 3,
                "mape": 0.21,
                "mean_err_kg_ha": 12.5,
                "median_err_kg_ha": -4.0,
                "mean_abs_err_kg_ha": 180.2,
                "median_abs_err_kg_ha": 150.0,
                "rmse_kg_ha": 240.7,
                "rrmse": 15.3,
                "r2_scikit": r2,
            }]
        ).to_csv(root / yr / TP / f"evaluation_{LEVEL}.csv", index=False)

    # all_predictions_*: what the pipeline writes by concatenating the per-year files.
    rows = []
    for yr, scale in ((PREV_YEAR, 0.8), (YEAR, 1.0)):
        for n in REGIONS:
            rows.append({
                "region": n,
                "mean_yield_kg_ha": YIELDS[n] * scale,
                "median_yield_kg_ha": YIELDS[n] * scale,
                "reported_mean_yield_kg_ha": YIELDS[n] * scale * 0.9,
                "total_production_ton": YIELDS[n] * scale * 0.01,
                "total_area_ha": 100.0,
                "year": int(yr),
            })
    pd.DataFrame(rows).to_csv(root / f"all_predictions_{STUDY}_{LEVEL}_{TP}.csv", index=False)

    return base, root


@pytest.fixture(scope="module")
def client(study_root, module_mocker=None):
    base, root = study_root
    import os

    os.environ["STUDY_DIR"] = str(base)
    import routers.maps as maps

    maps.studies_dir = str(base)

    # The aggregation level's shapefile comes from the run config.
    maps.get_run_config = lambda _dir, _study: {
        "eval_params": {
            "aggregation_levels": {
                LEVEL: {"shapefile": "levels.geojson", "name_column": "shapeName"},
                f"{LEVEL}_Extra": {"shapefile": "levels.geojson", "name_column": "shapeName"},
            }
        }
    }

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(maps.router, prefix="/api")
    return TestClient(app)


API = f"/api/studies/{STUDY}/results"


class TestManifest:
    def test_lists_years_levels_and_rasters(self, client):
        m = client.get(f"{API}/manifest").json()
        assert m["years"] == {PREV_YEAR: [TP], YEAR: [TP]}
        assert LEVEL in m["levels"] and f"{LEVEL}_Extra" in m["levels"]
        assert [r["kind"] for r in m["rasters"]] == ["yield"]
        assert m["default"]["year"] == YEAR

    def test_unknown_study_is_404(self, client):
        assert client.get("/api/studies/nosuch/results/manifest").status_code == 404


class TestStats:
    def test_values_match_the_source_csv_exactly(self, client, study_root):
        _base, root = study_root
        csv = pd.read_csv(root / YEAR / TP / f"agg_yield_estimates_{LEVEL}_{STUDY}_{YEAR}_{TP}.csv")
        served = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/stats").json()["regions"]
        assert set(served) == set(csv.region)
        for _, row in csv.iterrows():
            for col in ("mean_yield_kg_ha", "reported_mean_yield_kg_ha"):
                got, want = served[row.region][col], row[col]
                if pd.isna(want):
                    assert got is None, f"{row.region}.{col}: missing value must serialise as null"
                else:
                    assert got == want, f"{row.region}.{col}"

    def test_level_name_is_matched_exactly_not_by_prefix(self, client):
        """'ADM1' must not pick up 'ADM1_Extra' - the collision that broke reporting."""
        got = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/stats").json()["regions"]
        assert got["Alpha"]["mean_yield_kg_ha"] == YIELDS["Alpha"]
        extra = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}_Extra/stats").json()["regions"]
        assert extra["Alpha"]["mean_yield_kg_ha"] == 1.0

    def test_range_is_reported(self, client):
        r = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/stats").json()["range"]
        assert r["min"] == min(YIELDS.values()) and r["max"] == max(YIELDS.values())

    def test_missing_level_is_404(self, client):
        assert client.get(f"{API}/{YEAR}/{TP}/levels/Nope/stats").status_code == 404

    def test_missing_values_serialise_as_null_not_nan(self, client):
        """NaN is not valid JSON; letting one through fails the entire response."""
        r = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/stats")
        assert r.status_code == 200
        assert "NaN" not in r.text
        assert r.json()["regions"]["Gamma"]["reported_mean_yield_kg_ha"] is None

    def test_primary_level_drops_regions_that_were_never_simulated(self, client):
        """The primary CSV lists the whole source shapefile; all-NaN rows have no
        geometry on the map and would otherwise bloat every response."""
        r = client.get(f"{API}/{YEAR}/{TP}/levels/primary/stats")
        assert r.status_code == 200
        regions = r.json()["regions"]
        assert set(regions) == set(REGIONS)
        assert not any(k.startswith("unsimulated_") for k in regions)


class TestGeometry:
    def test_returns_one_feature_per_region_deduplicated(self, client):
        """The shapefile has two rows per region (one per year); the map needs one."""
        fc = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/geometry?zoom=8").json()
        names = [f["properties"]["region"] for f in fc["features"]]
        assert sorted(names) == sorted(REGIONS)

    def test_primary_level_uses_per_year_boundaries(self, client):
        fc = client.get(f"{API}/{YEAR}/{TP}/levels/primary/geometry?zoom=8").json()
        assert sorted(f["properties"]["region"] for f in fc["features"]) == sorted(REGIONS)

    def test_bbox_limits_what_is_sent(self, client):
        """Viewport filtering is what keeps a many-region study responsive."""
        fc = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/geometry?zoom=8&bbox=-0.1,-0.1,1.0,1.0").json()
        names = {f["properties"]["region"] for f in fc["features"]}
        assert "Alpha" in names and "Gamma" not in names

    def test_simplification_never_invents_or_drops_a_region(self, client):
        """Coarser zooms may simplify shapes but must not change region identity."""
        for zoom in (4, 8, 12):
            fc = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/geometry?zoom={zoom}").json()
            names = sorted(f["properties"]["region"] for f in fc["features"])
            assert names == sorted(REGIONS), f"zoom {zoom} changed the region set"

    def test_coarse_zoom_is_not_larger_than_fine_zoom(self, client):
        coarse = len(json.dumps(client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/geometry?zoom=3").json()))
        fine = len(json.dumps(client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/geometry?zoom=14").json()))
        assert coarse <= fine

    def test_etag_is_content_derived_and_stable(self, client):
        a = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/geometry?zoom=8")
        b = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/geometry?zoom=8")
        assert a.headers["etag"] == b.headers["etag"]

    def test_malformed_bbox_is_400(self, client):
        assert client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/geometry?bbox=1,2,3").status_code == 400
        assert client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/geometry?bbox=9,9,1,1").status_code == 400


class TestTiles:
    def test_returns_a_png(self, client):
        r = client.get(f"{API}/{YEAR}/{TP}/tiles/yield/8/128/127.png")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_tile_outside_bounds_is_transparent_not_an_error(self, client):
        r = client.get(f"{API}/{YEAR}/{TP}/tiles/yield/8/0/0.png")
        assert r.status_code == 200
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_unknown_raster_kind_is_404(self, client):
        assert client.get(f"{API}/{YEAR}/{TP}/tiles/nope/8/128/127.png").status_code == 404

    def test_bad_rescale_is_400(self, client):
        assert client.get(f"{API}/{YEAR}/{TP}/tiles/yield/8/128/127.png?rescale=abc").status_code == 400

    def test_tiles_are_cacheable(self, client):
        r = client.get(f"{API}/{YEAR}/{TP}/tiles/yield/8/128/127.png")
        assert "max-age" in r.headers.get("cache-control", "")


class TestRegionPlot:
    def test_reads_a_single_member_from_the_zip(self, client):
        r = client.get(f"{API}/{YEAR}/{TP}/regions/Alpha/plot.png")
        assert r.status_code == 200
        assert r.content.endswith(b"Alpha")

    def test_unknown_region_is_404(self, client):
        assert client.get(f"{API}/{YEAR}/{TP}/regions/Nobody/plot.png").status_code == 404

    def test_path_traversal_is_rejected(self, client):
        assert client.get(f"{API}/{YEAR}/{TP}/regions/..%2F..%2Fetc%2Fpasswd/plot.png").status_code in (400, 404)


class TestRasterStats:
    def test_reports_percentiles_for_the_legend(self, client):
        s = client.get(f"{API}/{YEAR}/{TP}/rasters/yield/stats").json()
        assert s["min"] <= s["p2"] <= s["p98"] <= s["max"]
        assert s["max"] > 0

    def test_unknown_kind_is_404(self, client):
        assert client.get(f"{API}/{YEAR}/{TP}/rasters/nope/stats").status_code == 404


class TestLegacyLevelNaming:
    """Older studies wrote agg_yield_estimates_* named after the aggregation shapefile's
    stem instead of the config key, so the level in the filename does not appear in
    eval_params.aggregation_levels. Geometry lookup must still resolve."""

    def test_level_named_after_the_shapefile_stem_resolves(self, client, monkeypatch):
        import routers.maps as maps

        monkeypatch.setattr(
            maps,
            "get_run_config",
            lambda _d, _s: {
                "eval_params": {"aggregation_levels": {"friendly_key": {"shapefile": "levels.geojson", "name_column": "shapeName"}}}
            },
        )
        maps._read_geometry.cache_clear()
        r = client.get(f"{API}/{YEAR}/{TP}/levels/levels/geometry?zoom=8")
        assert r.status_code == 200
        assert sorted(f["properties"]["region"] for f in r.json()["features"]) == sorted(REGIONS)


class TestCropmaskResolution:
    """The results map offers the study's own cropmask for the selected year; the manifest
    just names it, and the frontend draws it through the cropmask tiler."""

    def test_manifest_maps_year_to_cropmask_name(self, client, monkeypatch):
        import routers.maps as maps

        monkeypatch.setattr(
            maps,
            "get_run_config",
            lambda _d, _s: {"lai_params": {"crop_mask": {2024: "/data/cropmasks/kenya-maize-2024.tif"}}},
        )
        m = client.get(f"{API}/manifest").json()
        assert m["cropmasks"] == {"2024": "kenya-maize-2024"}

    def test_missing_config_yields_no_cropmasks_rather_than_failing(self, client, monkeypatch):
        import routers.maps as maps

        def boom(*_a, **_k):
            raise FileNotFoundError("no config")

        monkeypatch.setattr(maps, "get_run_config", boom)
        assert client.get(f"{API}/manifest").json()["cropmasks"] == {}


class TestManifestBounds:
    """Without an extent up front the map cannot frame the study, and a viewport-filtered
    geometry request from a default world view can legitimately return nothing - leaving a
    permanently blank map with no way to recover."""

    def test_manifest_carries_the_study_extent(self, client):
        b = client.get(f"{API}/manifest").json()["bounds"]
        assert b is not None
        west, south, east, north = b
        assert west < east and south < north
        # The fixture's regions span x 0..2.9, y 0..0.9
        assert west == pytest.approx(0, abs=0.01) and east == pytest.approx(2.9, abs=0.01)


class TestSeasons:
    """The imagery slider must span the window the yield was actually derived from,
    so it comes from lai_params.time_bounds rather than being guessed from the year."""

    def test_manifest_exposes_the_lai_window(self, client, monkeypatch):
        import routers.maps as maps

        monkeypatch.setattr(
            maps,
            "get_run_config",
            lambda _d, _s: {"lai_params": {"time_bounds": {2024: {"T-0": ["2024-02-01", "2024-10-01"]}}}},
        )
        seasons = client.get(f"{API}/manifest").json()["seasons"]
        assert seasons == {"2024": {"T-0": ["2024-02-01", "2024-10-01"]}}

    def test_malformed_or_missing_bounds_are_skipped(self, client, monkeypatch):
        import routers.maps as maps

        monkeypatch.setattr(
            maps,
            "get_run_config",
            lambda _d, _s: {"lai_params": {"time_bounds": {2024: {"T-0": ["2024-02-01"], "T-30": None}}}},
        )
        assert client.get(f"{API}/manifest").json()["seasons"] == {}


class TestLaiTimeseries:
    """The LAI curve is what the old prebuilt map showed on hover, and it is the evidence
    behind every yield number, so the map needs it back."""

    def _write(self, root):
        import pandas as pd

        rows = []
        for region in ("Alpha", "Beta"):
            for day, val in (("01/03/2024", 1.5), ("15/03/2024", 2.5), ("01/04/2024", None)):
                rows.append({"Date": day, "region": region, "LAI Median Adjusted": val})
        pd.DataFrame(rows).to_csv(
            root / YEAR / TP / f"agg_lai_timeseries_{LEVEL}_{STUDY}_{YEAR}_{TP}.csv", index=False
        )

    def test_returns_parallel_arrays_per_region(self, client, study_root):
        _base, root = study_root
        self._write(root)
        d = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/lai").json()
        assert d["dates"] == ["2024-03-01", "2024-03-15", "2024-04-01"]
        assert d["regions"]["Alpha"] == [1.5, 2.5, None]
        assert d["column"] == "LAI Median Adjusted"

    def test_dates_are_parsed_day_first(self, client, study_root):
        """01/03/2024 is 1 March; month-first parsing would reorder the whole season."""
        _base, root = study_root
        self._write(root)
        d = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/lai").json()
        assert d["dates"][0] == "2024-03-01"

    def test_missing_file_is_404(self, client):
        assert client.get(f"{API}/{YEAR}/{TP}/levels/primary/lai").status_code == 404


class TestSimplificationQuality:
    """Simplifying to a whole tile pixel collapsed county outlines into spikes. Shape, not
    just area, is what a reader follows along a boundary."""

    def test_shape_is_preserved_well_enough_to_recognise(self, client):
        import json

        coarse = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/geometry?zoom=7").json()
        fine = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/geometry?zoom=12").json()

        def verts(fc):
            n = 0
            for f in fc["features"]:
                n += len(json.dumps(f["geometry"]["coordinates"]).split("],"))
            return n

        # The fixture's boxes are already minimal, so the guard is that coarse never
        # degenerates below a closed ring and both zooms keep every region.
        assert len(coarse["features"]) == len(fine["features"]) == len(REGIONS)
        for f in coarse["features"]:
            ring = f["geometry"]["coordinates"][0]
            assert len(ring) >= 4, "a polygon ring needs at least 4 points to close"


class TestPrimaryLaiFallback:
    """No study has agg_lai_timeseries_primary_* - the pipeline rule excludes the primary
    level - but the same series exists per region, so the endpoint assembles it."""

    def test_primary_lai_is_built_from_per_region_stats(self, client, study_root):
        import pandas as pd

        _base, root = study_root
        for region in ("Alpha", "Beta"):
            d = root / YEAR / TP / region
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"Date": "01/03/2024", "LAI Median Adjusted": 1.1},
                    {"Date": "15/03/2024", "LAI Median Adjusted": 2.2},
                ]
            ).to_csv(d / f"{region}_LAI_STATS.csv", index=False)

        d = client.get(f"{API}/{YEAR}/{TP}/levels/primary/lai")
        assert d.status_code == 200
        body = d.json()
        assert body["dates"] == ["2024-03-01", "2024-03-15"]
        assert body["regions"]["Alpha"] == [1.1, 2.2]
        assert body["regions"]["Beta"] == [1.1, 2.2]


class TestArchivedRunIsSelfContained:
    """A snapshot must serve the whole interactive map on its own.

    This is what makes an archived run worth keeping: the numbers, the boundaries, the
    LAI curves and the pixel mosaic all have to come out of the snapshot, not out of
    whatever the live study happens to hold now. The snapshot here is built with the
    real packaging code and the real patterns file, so a pattern dropped from
    ``output_data_patterns.txt`` fails this test rather than silently producing archives
    that render half a map.
    """

    RUN_ID = "20260101_120000"

    @pytest.fixture(scope="class")
    def snapshot(self, study_root):
        import importlib.util

        _base, root = study_root
        # Needed by snapshot_aggregation_shapefiles, which reads the run config.
        (root / "config.yaml").write_text(
            "eval_params:\n"
            "  aggregation_levels:\n"
            f"    {LEVEL}:\n"
            "      shapefile: levels.geojson\n"
            "      name_column: shapeName\n"
        )

        pkg_dir = WEBAPP.parents[0] / "vercye_ops" / "reporting"
        spec = importlib.util.spec_from_file_location("_pkg", pkg_dir / "package_and_upload.py")
        pkg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pkg)

        patterns, _core = pkg.load_patterns(pkg_dir / "output_data_patterns.txt")
        dest = root / "run_results" / self.RUN_ID
        dest.mkdir(parents=True, exist_ok=True)
        pkg.copy_matches_to_snapshot(root, patterns, dest)
        pkg.snapshot_aggregation_shapefiles(root, dest)
        return dest

    @property
    def api(self):
        return f"/api/studies/{STUDY}/runs/{self.RUN_ID}/results"

    def test_manifest_offers_the_pixel_layer(self, client, snapshot):
        body = client.get(f"{self.api}/manifest").json()
        assert body["run_id"] == self.RUN_ID
        assert LEVEL in body["levels"]
        # Empty here would mean the 4326 mosaic never made it into the snapshot.
        assert [r["kind"] for r in body["rasters"]] == ["yield"]

    @pytest.mark.parametrize(
        "path",
        [
            f"/{YEAR}/{TP}/levels/{LEVEL}/geometry?zoom=8",
            f"/{YEAR}/{TP}/levels/{LEVEL}/stats",
            f"/{YEAR}/{TP}/levels/primary/lai",
            f"/{YEAR}/{TP}/rasters/yield/stats",
            f"/{YEAR}/{TP}/tiles/yield/8/128/127.png",
            f"/{YEAR}/{TP}/regions/Alpha/plot.png",
        ],
    )
    def test_every_map_endpoint_is_served_from_the_snapshot(self, client, snapshot, path):
        assert client.get(f"{self.api}{path}").status_code == 200

    def test_values_match_the_live_study(self, client, snapshot):
        live = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/stats").json()
        archived = client.get(f"{self.api}/{YEAR}/{TP}/levels/{LEVEL}/stats").json()
        assert archived["regions"] == live["regions"]

    def test_geometry_comes_from_the_snapshots_own_shapefile(self, client, snapshot):
        assert (snapshot / "aggregation_shapefiles" / "levels.geojson").is_file()
        live = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/geometry?zoom=8").json()
        archived = client.get(f"{self.api}/{YEAR}/{TP}/levels/{LEVEL}/geometry?zoom=8").json()
        assert sorted(f["properties"]["region"] for f in archived["features"]) == sorted(
            f["properties"]["region"] for f in live["features"]
        )

    def test_a_snapshot_without_a_mosaic_does_not_borrow_the_live_one(self, client, study_root):
        """Older snapshots predate the 4326 mosaic being packaged. Those must report no
        pixel layer - serving the live study's mosaic under an archived run's label would
        show the wrong run's pixels with no way to tell."""
        _base, root = study_root
        bare = root / "run_results" / "20250101_000000"
        (bare / YEAR / TP).mkdir(parents=True, exist_ok=True)
        api = f"/api/studies/{STUDY}/runs/20250101_000000/results"
        assert client.get(f"{api}/manifest").json()["rasters"] == []
        assert client.get(f"{api}/{YEAR}/{TP}/tiles/yield/8/128/127.png").status_code == 404
        assert client.get(f"{api}/{YEAR}/{TP}/rasters/yield/stats").status_code == 404


class TestPipelineMetrics:
    """Accuracy numbers are reported, never recomputed.

    The browser was deriving R2/RMSE/bias from the stats JSON. It happened to agree, but a
    panel that computes its own statistics can drift from the PDF reports silently, and it
    had no way to produce MAPE, relative RMSE or the error quantiles at all. These tests
    pin the API's output to the pipeline's own evaluation_{level}.csv.
    """

    def test_evaluation_matches_the_pipeline_csv(self, client, study_root):
        _base, root = study_root
        csv = pd.read_csv(root / YEAR / TP / f"evaluation_{LEVEL}.csv").iloc[0]
        body = client.get(f"{API}/{YEAR}/{TP}/levels/{LEVEL}/evaluation").json()
        assert body["metrics"]["r2_scikit"] == pytest.approx(csv.r2_scikit)
        assert body["metrics"]["rmse_kg_ha"] == pytest.approx(csv.rmse_kg_ha)
        assert body["metrics"]["rrmse"] == pytest.approx(csv.rrmse)
        assert body["metrics"]["mape"] == pytest.approx(csv.mape)

    def test_level_without_reference_data_reports_no_metrics(self, client):
        """The pipeline writes no evaluation file for a level with no ground truth, and
        that is a legitimate state, not an error - the panel just hides the block."""
        body = client.get(f"{API}/{YEAR}/{TP}/levels/primary/evaluation").json()
        assert body["metrics"] is None

    def test_multiyear_carries_one_metric_row_per_year(self, client, study_root):
        _base, root = study_root
        body = client.get(f"/api/studies/{STUDY}/results/multiyear/{TP}/levels/{LEVEL}").json()
        assert body["years"] == [PREV_YEAR, YEAR]
        by_year = {m["year"]: m for m in body["metrics"]}
        for yr in (PREV_YEAR, YEAR):
            expected = pd.read_csv(root / yr / TP / f"evaluation_{LEVEL}.csv").iloc[0]
            assert by_year[yr]["r2_scikit"] == pytest.approx(expected.r2_scikit)

    def test_multiyear_region_series_come_from_all_predictions(self, client, study_root):
        _base, root = study_root
        csv = pd.read_csv(root / f"all_predictions_{STUDY}_{LEVEL}_{TP}.csv")
        body = client.get(f"/api/studies/{STUDY}/results/multiyear/{TP}/levels/{LEVEL}").json()
        assert body["source"] == f"all_predictions_{STUDY}_{LEVEL}_{TP}.csv"
        for region, series in body["regions"].items():
            # Sorted by year, and each value is the file's value untouched.
            assert [r["year"] for r in series] == sorted(r["year"] for r in series)
            for row in series:
                src = csv[(csv.region == region) & (csv.year == int(row["year"]))].iloc[0]
                assert row["predicted"] == pytest.approx(src.mean_yield_kg_ha)
                assert row["reported"] == pytest.approx(src.reported_mean_yield_kg_ha)


class TestSharedColourScales:
    """One domain per metric for the whole study.

    A domain recomputed from whatever is on screen meant a colour changed meaning every
    time the year or the admin level changed, which makes two screenshots of the same map
    incomparable. The manifest carries one range per metric instead.
    """

    def test_manifest_carries_a_range_per_metric(self, client):
        scales = client.get(f"{API}/manifest").json()["scales"]
        assert "mean_yield_kg_ha" in scales
        assert scales["mean_yield_kg_ha"]["max"] > scales["mean_yield_kg_ha"]["min"]

    def test_the_range_spans_every_year_not_just_the_latest(self, client, study_root):
        """The prior year's yields are 0.8x the latest year's, so a domain built from one
        year alone would start above the other year's minimum."""
        _base, root = study_root
        csv = pd.read_csv(root / f"all_predictions_{STUDY}_{LEVEL}_{TP}.csv")
        scales = client.get(f"{API}/manifest").json()["scales"]
        assert scales["mean_yield_kg_ha"]["min"] == pytest.approx(csv.mean_yield_kg_ha.min())

    def test_diverging_metrics_are_symmetric_about_zero(self, client):
        scales = client.get(f"{API}/manifest").json()["scales"]
        rel = scales.get("rel_error")
        if rel:
            assert rel["min"] == pytest.approx(-rel["max"])

    def test_metrics_by_level_says_where_a_value_exists(self, client):
        """The toolbar shows the same list everywhere and disables what a level lacks, so
        it needs to know which levels do carry each metric."""
        by_level = client.get(f"{API}/manifest").json()["metrics_by_level"]
        assert "mean_yield_kg_ha" in by_level[LEVEL]
        # Derived error metrics exist wherever both sides of the comparison do.
        assert "abs_error" in by_level[LEVEL] and "rel_error" in by_level[LEVEL]


class TestStudyWithoutReferenceData:
    """A study with no ground truth anywhere must still serve its whole map.

    Ethiopia has no reported yields at any level, so its all_predictions files carry no
    `reported_mean_yield_kg_ha` column. The shared-scale computation reached for that
    column with `df.get(...)`, which returns None, and `pd.to_numeric(None)` is a NaN
    *scalar* rather than None - so the "is not None" guard passed and the next line called
    .replace on a float. That raised inside the manifest, which is the one request every
    view of the study depends on, so the entire map 500'd.
    """

    @pytest.fixture(scope="class")
    def noref_client(self, tmp_path_factory):
        import os
        import sys

        base = tmp_path_factory.mktemp("noref_studies")
        study = "norefstudy"
        root = base / study / study
        ytp = root / YEAR / TP
        ytp.mkdir(parents=True)
        (root / "aggregation_shapefiles").mkdir()

        gpd.GeoDataFrame(
            [{"shapeName": n, "geometry": box(x, y, x + 0.9, y + 0.9)} for n, (x, y) in REGIONS.items()],
            crs="EPSG:4326",
        ).to_file(root / "aggregation_shapefiles" / "levels.geojson", driver="GeoJSON")

        # No reported_mean_yield_kg_ha column anywhere - this is the whole point.
        rows = [{"region": n, "mean_yield_kg_ha": YIELDS[n], "total_production_ton": 1.0} for n in REGIONS]
        pd.DataFrame(rows).to_csv(ytp / f"agg_yield_estimates_{LEVEL}_{study}_{YEAR}_{TP}.csv", index=False)
        pd.DataFrame([{**r, "year": int(YEAR)} for r in rows]).to_csv(
            root / f"all_predictions_{study}_{LEVEL}_{TP}.csv", index=False
        )

        sys.path.insert(0, str(WEBAPP))
        os.environ["STUDY_DIR"] = str(base)
        import routers.maps as maps

        saved_dir, saved_cfg = maps.studies_dir, maps.get_run_config
        maps.studies_dir = str(base)
        maps.get_run_config = lambda _d, _s: {
            "eval_params": {"aggregation_levels": {LEVEL: {"shapefile": "levels.geojson", "name_column": "shapeName"}}}
        }

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(maps.router, prefix="/api")
        yield TestClient(app), study
        maps.studies_dir, maps.get_run_config = saved_dir, saved_cfg

    def test_manifest_is_served(self, noref_client):
        client, study = noref_client
        r = client.get(f"/api/studies/{study}/results/manifest")
        assert r.status_code == 200, r.text
        assert LEVEL in r.json()["levels"]

    def test_error_metrics_are_simply_absent(self, noref_client):
        """Not derivable without a reference, so they must not be offered - and the
        frontend greys them out on the strength of exactly this."""
        client, study = noref_client
        m = client.get(f"/api/studies/{study}/results/manifest").json()
        assert "rel_error" not in m["scales"] and "abs_error" not in m["scales"]
        assert "rel_error" not in m["metrics_by_level"].get(LEVEL, [])
        # The values that do exist are still scaled.
        assert "mean_yield_kg_ha" in m["scales"]

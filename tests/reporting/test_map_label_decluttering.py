"""Region labels on the report map must be dropped where they cannot fit.

Previously labelling was all-or-nothing (`if len(merged) < 70`), so a map with a
few large regions and many small ones rendered an unreadable pile of overlapping
text over the small ones.
"""

import importlib.util
from pathlib import Path

import geopandas as gpd
import matplotlib
import pytest
from shapely.geometry import MultiPolygon, Polygon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTING = Path(__file__).resolve().parents[2] / "vercye_ops" / "reporting"


def _load():
    spec = importlib.util.spec_from_file_location("gafr", REPORTING / "generate_aggregated_final_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _square(x, y, size):
    return Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])


def _frame(geoms, names):
    return gpd.GeoDataFrame(
        {"region": names, "mean_yield_kg_ha": [1500.0] * len(geoms), "geometry": geoms},
        crs="EPSG:4326",
    )


@pytest.fixture
def ax_and_mod():
    mod = _load()
    fig, ax = plt.subplots(figsize=(12, 8))
    yield ax, mod
    plt.close(fig)


class TestLabelFitting:
    def test_tiny_regions_are_not_labelled(self, ax_and_mod):
        ax, mod = ax_and_mod
        # one big region plus many pin-sized ones crowded into a corner
        geoms = [_square(0, 0, 10)] + [_square(10.1 + 0.02 * i, 0, 0.01) for i in range(20)]
        names = ["Big"] + [f"Tiny{i}" for i in range(20)]
        merged = _frame(geoms, names)
        merged.plot(ax=ax, column="mean_yield_kg_ha")
        cmap = plt.get_cmap("viridis")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=3000)

        placed, skipped = mod._place_region_labels(ax, merged, cmap, norm)
        assert placed >= 1, "the large region should keep its label"
        assert skipped >= 15, "pin-sized regions should be dropped, not overplotted"
        assert placed + skipped == len(merged)

    def test_large_regions_keep_labels(self, ax_and_mod):
        ax, mod = ax_and_mod
        merged = _frame([_square(0, 0, 10), _square(11, 0, 10)], ["Alpha", "Beta"])
        merged.plot(ax=ax, column="mean_yield_kg_ha")
        placed, skipped = mod._place_region_labels(
            ax, merged, plt.get_cmap("viridis"), matplotlib.colors.Normalize(0, 3000)
        )
        assert placed == 2 and skipped == 0

    def test_label_sits_inside_concave_geometry(self, ax_and_mod):
        """centroid can fall outside a concave/multipart shape; representative_point cannot."""
        ax, mod = ax_and_mod
        horseshoe = Polygon([(0, 0), (10, 0), (10, 10), (7, 10), (7, 3), (3, 3), (3, 10), (0, 10)])
        merged = _frame([horseshoe], ["Horseshoe"])
        merged.plot(ax=ax, column="mean_yield_kg_ha")
        placed, _ = mod._place_region_labels(ax, merged, plt.get_cmap("viridis"), matplotlib.colors.Normalize(0, 3000))
        assert placed == 1
        from shapely.geometry import Point

        x, y = ax.texts[0].get_position()
        assert horseshoe.covers(Point(x, y)), "label was placed outside the polygon"
        assert not horseshoe.covers(horseshoe.centroid), "test is only meaningful if the centroid falls outside"

    def test_multipart_geometry_does_not_crash(self, ax_and_mod):
        ax, mod = ax_and_mod
        mp = MultiPolygon([_square(0, 0, 5), _square(20, 20, 5)])
        merged = _frame([mp], ["Island"])
        merged.plot(ax=ax, column="mean_yield_kg_ha")
        placed, skipped = mod._place_region_labels(
            ax, merged, plt.get_cmap("viridis"), matplotlib.colors.Normalize(0, 3000)
        )
        assert placed + skipped == 1

    def test_over_cap_skips_all_labels(self, ax_and_mod):
        ax, mod = ax_and_mod
        geoms = [_square(i, 0, 0.5) for i in range(12)]
        merged = _frame(geoms, [f"R{i}" for i in range(12)])
        merged.plot(ax=ax, column="mean_yield_kg_ha")
        placed, skipped = mod._place_region_labels(
            ax, merged, plt.get_cmap("viridis"), matplotlib.colors.Normalize(0, 3000), max_regions=5
        )
        assert placed == 0 and skipped == 12

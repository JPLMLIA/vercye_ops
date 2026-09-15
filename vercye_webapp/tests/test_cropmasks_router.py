"""Tests for cropmask metadata, overviews and tiles.

The season metadata is what bounds the Sentinel-2 month slider, so the rules that matter
are: it is all-or-nothing, it is validated, and its absence degrades gracefully (legacy
masks keep working, they just get basemaps only).
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

WEBAPP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEBAPP))


def _mask_bytes(values=(0, 1)) -> bytes:
    """A small binary GeoTIFF, as an upload would arrive."""
    data = np.zeros((1, 64, 64), dtype="uint8")
    data[0, 10:40, 10:40] = 1
    if values != (0, 1):
        data[0, 0, 0] = values[-1]
    buf = Path("/tmp") / f"_cm_{values[-1]}.tif"
    with rasterio.open(
        buf, "w", driver="GTiff", height=64, width=64, count=1, dtype="uint8",
        crs="EPSG:4326", transform=from_bounds(34, -1, 36, 1, 64, 64),
    ) as dst:
        dst.write(data)
    return buf.read_bytes()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import routers.cropmasks as cm

    monkeypatch.setattr(cm, "cropmasks_dir", str(tmp_path))
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(cm.router, prefix="/api")
    return TestClient(app)


def _upload(client, name="mask", **form):
    return client.post(
        f"/api/cropmasks/{name}",
        files={"cropmask_file": (f"{name}.tif", io.BytesIO(_mask_bytes()), "image/tiff")},
        data={k: str(v) for k, v in form.items()},
    )


class TestUpload:
    def test_upload_without_season_still_works(self, client):
        """Legacy behaviour must be preserved: metadata is optional."""
        r = _upload(client)
        assert r.status_code == 200
        listed = client.get("/api/cropmasks").json()
        assert listed[0]["year"] is None and listed[0]["season_start"] is None

    def test_upload_with_season_stores_it(self, client):
        r = _upload(client, year=2024, season_start="2024-03-01", season_end="2024-09-30")
        assert r.status_code == 200
        row = client.get("/api/cropmasks").json()[0]
        assert row["year"] == 2024
        assert row["season_start"] == "2024-03-01"
        assert row["season_end"] == "2024-09-30"

    def test_partial_season_is_rejected(self, client):
        """Half a range would produce an uninterpretable slider."""
        r = _upload(client, year=2024, season_start="2024-03-01")
        assert r.status_code == 400

    def test_reversed_season_is_rejected(self, client):
        r = _upload(client, year=2024, season_start="2024-09-30", season_end="2024-03-01")
        assert r.status_code == 400

    def test_bad_date_format_is_rejected(self, client):
        r = _upload(client, year=2024, season_start="March 2024", season_end="2024-09-30")
        assert r.status_code == 400

    def test_duplicate_name_is_409(self, client):
        assert _upload(client).status_code == 200
        assert _upload(client).status_code == 409

    def test_non_binary_raster_is_rejected(self, client):
        r = client.post(
            "/api/cropmasks/bad",
            files={"cropmask_file": ("bad.tif", io.BytesIO(_mask_bytes(values=(0, 7))), "image/tiff")},
        )
        assert r.status_code == 400
        assert "7" in r.json()["detail"]


class TestMetadataUpdate:
    def test_legacy_mask_can_gain_a_season(self, client):
        _upload(client)
        r = client.put(
            "/api/cropmasks/mask/metadata",
            data={"year": "2023", "season_start": "2023-03-01", "season_end": "2023-08-31"},
        )
        assert r.status_code == 200
        assert client.get("/api/cropmasks/mask/info").json()["year"] == 2023

    def test_metadata_can_be_cleared(self, client):
        _upload(client, year=2024, season_start="2024-03-01", season_end="2024-09-30")
        client.put("/api/cropmasks/mask/metadata", data={})
        assert client.get("/api/cropmasks/mask/info").json()["year"] is None

    def test_unknown_mask_is_404(self, client):
        assert client.put("/api/cropmasks/nope/metadata", data={}).status_code == 404


class TestInfoAndTiles:
    def test_info_reports_bounds_in_wgs84(self, client):
        _upload(client)
        info = client.get("/api/cropmasks/mask/info").json()
        assert info["bounds"] == pytest.approx([34, -1, 36, 1], abs=1e-6)
        assert info["width"] == 64

    def test_tile_renders_png(self, client):
        _upload(client)
        r = client.get("/api/cropmasks/mask/tiles/8/145/127.png")
        assert r.status_code == 200
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_tile_outside_bounds_is_transparent(self, client):
        _upload(client)
        r = client.get("/api/cropmasks/mask/tiles/8/0/0.png")
        assert r.status_code == 200
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_unknown_mask_tile_is_404(self, client):
        assert client.get("/api/cropmasks/nope/tiles/8/145/127.png").status_code == 404

    def test_path_traversal_is_rejected(self, client):
        assert client.get("/api/cropmasks/..%2F..%2Fetc%2Fpasswd/info").status_code in (400, 404)


class TestOverviews:
    def test_overviews_are_external_so_the_original_is_untouched(self, client, tmp_path):
        """The pipeline reads these files; building overviews must not rewrite them."""
        _upload(client)
        tif = tmp_path / "mask.tif"
        before = tif.read_bytes()
        import routers.cropmasks as cm

        cm.build_overviews(tif)
        assert tif.read_bytes() == before, "the source raster must not be modified"

    def test_no_stale_lock_is_left_behind(self, client, tmp_path):
        _upload(client)
        assert not list(tmp_path.glob("*.building"))


class TestPreparation:
    """A mask without overviews must not block tile requests: a single low-zoom tile on a
    109k x 130k mask measured 10.8 s, and Leaflet requests many at once."""

    def test_prepare_reports_ready_once_overviews_exist(self, client):
        _upload(client)  # upload builds them
        r = client.post("/api/cropmasks/mask/prepare")
        assert r.status_code == 200 and r.json()["ready"] is True

    def test_prepare_on_unknown_mask_is_404(self, client):
        assert client.post("/api/cropmasks/nope/prepare").status_code == 404

    def test_low_zoom_tile_is_a_placeholder_while_overviews_are_missing(self, client, tmp_path, monkeypatch):
        _upload(client)
        import routers.cropmasks as cm

        monkeypatch.setattr(cm, "_has_overviews", lambda _p: False)
        r = client.get("/api/cropmasks/mask/tiles/3/4/4.png")
        assert r.status_code == 200
        assert r.headers.get("X-Cropmask-Status") == "preparing"
        assert r.headers["cache-control"] == "no-store"

    def test_high_zoom_tile_is_served_even_without_overviews(self, client, monkeypatch):
        """A direct read is cheap when the tile covers little ground."""
        _upload(client)
        import routers.cropmasks as cm

        monkeypatch.setattr(cm, "_has_overviews", lambda _p: False)
        r = client.get("/api/cropmasks/mask/tiles/12/2328/2047.png")
        assert r.status_code == 200
        assert r.headers.get("X-Cropmask-Status") is None

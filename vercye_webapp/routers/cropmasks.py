import json
import os
import shutil
import subprocess
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio as rio
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, Response, UploadFile

from vercye_ops.utils.env_utils import read_cropmasks_dir_from_env

router = APIRouter(
    prefix="/cropmasks",
    tags=["cropmasks"],
)

cropmasks_dir = read_cropmasks_dir_from_env()

# Season metadata lives beside the raster rather than inside it, so uploading it neither
# rewrites the GeoTIFF nor invalidates anything that already points at the file. Masks
# uploaded before this existed simply have no sidecar - they stay usable, they just get
# basemaps without the Sentinel-2 time slider.
META_SUFFIX = ".meta.json"


def _tif_path(cropmask_id: str) -> Path:
    if "/" in cropmask_id or "\\" in cropmask_id or ".." in cropmask_id:
        raise HTTPException(status_code=400, detail="Invalid cropmask id.")
    return Path(cropmasks_dir) / f"{cropmask_id}.tif"


def _meta_path(cropmask_id: str) -> Path:
    return Path(cropmasks_dir) / f"{cropmask_id}{META_SUFFIX}"


def _read_meta(cropmask_id: str) -> Optional[dict]:
    p = _meta_path(cropmask_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _validate_season(year: Optional[int], start: Optional[str], end: Optional[str]) -> Optional[dict]:
    """All three or none. The season bounds drive the imagery time slider, so a partial
    or reversed range would produce a slider that cannot be interpreted."""
    provided = [v for v in (year, start, end) if v not in (None, "")]
    if not provided:
        return None
    if len(provided) != 3:
        raise HTTPException(status_code=400, detail="Provide year, season_start and season_end together, or none.")
    try:
        s, e = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    except ValueError:
        raise HTTPException(status_code=400, detail="season_start and season_end must be YYYY-MM-DD.")
    if e < s:
        raise HTTPException(status_code=400, detail="season_end must not precede season_start.")
    return {"year": int(year), "season_start": s.isoformat(), "season_end": e.isoformat()}


def _has_overviews(tif: Path) -> bool:
    if (tif.parent / f"{tif.name}.ovr").is_file():
        return True
    try:
        with rio.open(tif) as src:
            return bool(src.overviews(1))
    except Exception:
        return False


def build_overviews(tif: Path) -> bool:
    """Create external overviews (a .ovr sidecar) so low-zoom tiles are cheap.

    Uploaded cropmasks are single-resolution: a national 109k x 130k mask makes a zoom-6
    tile read the whole country (measured 1609 ms, versus 9 ms with overviews). `-ro`
    keeps them external, so the file the pipeline reads is never modified.
    """
    if _has_overviews(tif):
        return True
    lock = tif.parent / f"{tif.name}.ovr.building"
    if lock.exists():
        return False  # another request is already building them
    try:
        lock.touch()
        subprocess.run(
            [
                "gdaladdo", "-ro", "-q",
                "--config", "COMPRESS_OVERVIEW", "LZW",
                "--config", "GDAL_NUM_THREADS", "ALL_CPUS",
                "-r", "average", str(tif), "2", "4", "8", "16", "32", "64", "128", "256",
            ],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    finally:
        lock.unlink(missing_ok=True)


@router.get("")
def get_all_cropmasks():
    if not os.path.exists(cropmasks_dir):
        return []

    out = []
    for f in sorted(os.listdir(cropmasks_dir)):
        if not f.endswith(".tif"):
            continue
        cid = f[: -len(".tif")]
        meta = _read_meta(cid)
        tif = Path(cropmasks_dir) / f
        out.append(
            {
                "id": f,
                "name": cid,
                "size_bytes": tif.stat().st_size if tif.is_file() else None,
                # Present only for masks uploaded with season metadata; the viewer uses it
                # to bound the Sentinel-2 month slider and falls back to basemaps without it.
                "year": (meta or {}).get("year"),
                "season_start": (meta or {}).get("season_start"),
                "season_end": (meta or {}).get("season_end"),
                "has_overviews": _has_overviews(tif),
            }
        )
    return out


@router.get("/{cropmask_id}/info")
def get_cropmask_info(cropmask_id: str):
    tif = _tif_path(cropmask_id)
    if not tif.is_file():
        raise HTTPException(status_code=404, detail="Cropmask not found.")
    meta = _read_meta(cropmask_id) or {}
    with rio.open(tif) as src:
        b = src.bounds
        if src.crs and src.crs.to_epsg() != 4326:
            from rasterio.warp import transform_bounds

            b = transform_bounds(src.crs, "EPSG:4326", *b)
        info = {"width": src.width, "height": src.height, "crs": str(src.crs)}
    return {
        "name": cropmask_id,
        "bounds": [b[0], b[1], b[2], b[3]],
        **info,
        "year": meta.get("year"),
        "season_start": meta.get("season_start"),
        "season_end": meta.get("season_end"),
        "has_overviews": _has_overviews(tif),
    }


# Below this zoom a tile covers enough ground that reading a mask without overviews means
# scanning most of the raster: measured 10.8 s for a single z5 tile on a 109k x 130k mask,
# versus 15 ms once overviews exist. Above it, a direct read is fine (~18 ms).
MIN_ZOOM_WITHOUT_OVERVIEWS = 10


@router.post("/{cropmask_id}/prepare")
def prepare_cropmask(cropmask_id: str, background: BackgroundTasks):
    """Kick off overview building so the viewer can wait with a message instead of hanging.

    Idempotent: returns ready immediately when overviews already exist.
    """
    tif = _tif_path(cropmask_id)
    if not tif.is_file():
        raise HTTPException(status_code=404, detail="Cropmask not found.")
    if _has_overviews(tif):
        return {"ready": True, "building": False}
    background.add_task(build_overviews, tif)
    return {"ready": False, "building": True}


# Cropland is drawn in magenta rather than the obvious green. The mask is meant to be read
# *against* its backdrop - Esri satellite, Sentinel-2 visual, or a pale paper basemap - and
# green cropland over green vegetation is the one combination where the layer disappears
# exactly where it matters. Magenta occurs in no natural land cover, so it separates from
# vegetation, bare soil, water and light grey paper alike.
CROPLAND_COLOR = "#ff2bbf"


@router.get("/{cropmask_id}/tiles/{z}/{x}/{y}.png")
def get_cropmask_tile(
    cropmask_id: str, z: int, x: int, y: int, background: BackgroundTasks, color: str = Query(CROPLAND_COLOR)
):
    """Cropland pixels in a solid colour, non-cropland fully transparent."""
    tif = _tif_path(cropmask_id)
    if not tif.is_file():
        raise HTTPException(status_code=404, detail="Cropmask not found.")

    if not _has_overviews(tif):
        # Build in the background rather than blocking this request - Leaflet asks for a
        # dozen tiles at once and each would otherwise wait on the same slow scan.
        background.add_task(build_overviews, tif)
        if z < MIN_ZOOM_WITHOUT_OVERVIEWS:
            return Response(
                content=_blank_png(),
                media_type="image/png",
                # Do not cache: the real tile becomes available once overviews land.
                headers={"Cache-Control": "no-store", "X-Cropmask-Status": "preparing"},
            )

    from rio_tiler.errors import TileOutsideBounds
    from rio_tiler.io import Reader

    try:
        with Reader(str(tif)) as src:
            img = src.tile(x, y, z, tilesize=256, resampling_method="average")
    except TileOutsideBounds:
        return Response(content=_blank_png(), media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})

    png = img.render(img_format="PNG", colormap=_mask_colormap(color))
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@lru_cache(maxsize=8)
def _mask_colormap(color: str) -> dict:
    """0 (and anything that averaged down to nothing) transparent; cropland opaque.

    Overviews are built with `average`, so a downsampled pixel holds the *fraction* of
    cropland beneath it. Carrying that through as opacity keeps sparse cropland visible
    when zoomed out instead of disappearing to a hard 0/1 threshold.
    """
    c = color.lstrip("#")
    r, g, b = (int(c[i : i + 2], 16) for i in (0, 2, 4))
    return {i: (r, g, b, 0 if i == 0 else min(255, 60 + i)) for i in range(256)}


@lru_cache(maxsize=1)
def _blank_png() -> bytes:
    from rio_tiler.models import ImageData

    arr = np.ma.MaskedArray(np.zeros((1, 256, 256), dtype="uint8"), mask=np.ones((1, 256, 256), dtype=bool))
    return ImageData(arr).render(img_format="PNG")


@router.post("/{cropmask_id}")
def upload_cropmask(
    cropmask_id: str,
    cropmask_file: UploadFile = File(...),
    year: Optional[int] = Form(None),
    season_start: Optional[str] = Form(None),
    season_end: Optional[str] = Form(None),
):
    storage_path = _tif_path(cropmask_id)
    meta = _validate_season(year, season_start, season_end)

    if storage_path.exists():
        raise HTTPException(status_code=409, detail="A cropmask with this name already exists")

    # Save temporarily for validation
    tmp_path = str(storage_path) + ".tmp"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(cropmask_file.file, f)

    # Validate raster content to be binary (0/1)
    try:
        with rio.open(tmp_path) as src:
            arr = src.read(1, masked=True)  # read first band
            unique_vals = np.unique(arr.compressed())  # ignore nodata

            if not np.all(np.isin(unique_vals, [0, 1])):
                os.remove(tmp_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid cropmask values found: {unique_vals.tolist()}. Only 0 and 1 allowed.",
                )

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"Invalid GeoTIFF: {str(e)}")

    os.rename(tmp_path, storage_path)
    if meta:
        _meta_path(cropmask_id).write_text(json.dumps(meta, indent=2))
    # Cheap for typical masks (~1 s for a 49k x 49k one) and makes the viewer usable
    # immediately; a failure here is not fatal, tiles just build them on first request.
    build_overviews(storage_path)
    return {"message": f"Cropmask {cropmask_id} uploaded successfully", **(meta or {})}


@router.put("/{cropmask_id}/metadata")
def set_cropmask_metadata(
    cropmask_id: str,
    year: Optional[int] = Form(None),
    season_start: Optional[str] = Form(None),
    season_end: Optional[str] = Form(None),
):
    """Attach (or clear) season metadata on an existing mask, so legacy uploads can gain
    a time slider without being re-uploaded."""
    if not _tif_path(cropmask_id).is_file():
        raise HTTPException(status_code=404, detail="Cropmask not found.")
    meta = _validate_season(year, season_start, season_end)
    if meta:
        _meta_path(cropmask_id).write_text(json.dumps(meta, indent=2))
    else:
        _meta_path(cropmask_id).unlink(missing_ok=True)
    return {"message": "Metadata updated", **(meta or {})}

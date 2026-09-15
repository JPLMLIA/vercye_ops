"""Data endpoints backing the interactive results map.

The pipeline also emits a self-contained ``interactive_map_*.zip`` (a single HTML with
every level's GeoJSON inlined plus one PNG per region). That stays available as the
legacy view, but it does not scale: Ukraine's is 10 MiB of embedded geometry plus
189 MiB of PNGs, all delivered before the first paint, and one file per year/timepoint
so switching means reloading everything.

This router serves the same underlying numbers as small, separately cacheable pieces:

  manifest   - what exists (years, timepoints, levels, raster kinds, value ranges)
  geometry   - polygons only, simplified for the requested zoom and clipped to the
               viewport, with a content-derived ETag so switching year reuses the
               browser's copy (region geometry is identical across years in practice,
               but the ETag makes that an observation rather than an assumption)
  stats      - one small JSON of per-region values; this is all that changes when the
               user switches year
  tiles      - XYZ PNG tiles rendered by rio-tiler straight from the COG's internal
               overviews, so a 326 MiB / 46k x 46k mosaic costs ~10 ms per tile
  plot       - a single region's report PNG, read out of the zip on click

Correctness note: everything here reads the exact same files the prebuilt map and the
PDF reports do - ``agg_yield_estimates_{level}_*.csv`` for numbers and the boundaries
GeoJSON / aggregation shapefiles for shapes. No values are recomputed.
"""

import hashlib
import json
import math
import os
import zipfile
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Response
from models import RunID, StudyID

from vercye_ops.utils.env_utils import get_run_config, read_studies_dir_from_env

router = APIRouter(prefix="/studies", tags=["maps"])

studies_dir = read_studies_dir_from_env()

# Raster layers offered by the map. Keys are the API's stable names; values are the
# glob for the EPSG:4326 COG the pipeline writes (the "_projected_" equal-area twin is
# for area maths, not for display).
RASTER_KINDS = {
    "yield": ("yield_mosaic_4326_", "Yield (LAI-converted)"),
    "apsim_yield": ("apsim_yield_mosaic_4326_", "Yield (APSIM only)"),
}

# Level name used for the per-region simulation level, which unlike the aggregation
# levels has its geometry written per year/timepoint rather than in a shapefile.
PRIMARY_LEVEL = "primary"

# Web-mercator tile pixels are 256 wide; one pixel at zoom z spans this many degrees.
def _deg_per_pixel(zoom: int) -> float:
    return 360.0 / (256.0 * (2**zoom))


# Simplification tolerance as a fraction of a tile pixel. A full pixel is too coarse in
# practice - it took a 898-vertex county down to 20 and the outline visibly collapsed into
# spikes - because shapes are drawn on a retina canvas and the eye follows a boundary's
# shape, not just its area. A quarter pixel keeps ~80 vertices with 0.13% area drift.
SIMPLIFY_PIXEL_FRACTION = 0.25

# Features whose whole extent is under this many pixels cannot render distinguishably.
DROP_BELOW_PIXELS = 0.5


# --------------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------------- #


def _study_root(study_id: str) -> Path:
    return Path(studies_dir) / study_id / study_id


def _run_root(study_id: str, run_id: str) -> Path:
    root = _study_root(study_id) / "run_results"
    run_dir = root / run_id
    try:
        run_dir.resolve(strict=True).relative_to(root.resolve())
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.")
    return run_dir


def _resolve_root(study_id: str, run_id: Optional[str]) -> Path:
    root = _run_root(study_id, run_id) if run_id else _study_root(study_id)
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"No results for study '{study_id}'.")
    return root


def _fingerprint(path: Path) -> str:
    """Cheap identity for a file, used for cache keys and ETags."""
    st = path.stat()
    return f"{st.st_size}-{st.st_mtime_ns}"


def _content_etag(path: Path) -> str:
    """Content hash, so two years whose geometry is byte-identical share an ETag and
    the browser can reuse its copy when the user switches year."""
    return _hash_file(str(path), _fingerprint(path))


@lru_cache(maxsize=64)
def _hash_file(path: str, _fp: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------------- #


def _extract_level(filename: str, study_id: str, year: str, timepoint: str) -> Optional[str]:
    """Level name out of agg_yield_estimates_{level}_{study}_{year}_{tp}.csv.

    Both the level and the study id may contain underscores, so strip the known prefix
    and the exact suffix rather than splitting on '_' - a plain
    f"...{level}_*" glob lets one level swallow another whose name merely starts with
    it (this is the ADM1 / ADM1_ThreeCounties collision fixed in the reporting code).
    """
    base = os.path.basename(filename)
    prefix, suffix = "agg_yield_estimates_", f"_{study_id}_{year}_{timepoint}.csv"
    if not base.startswith(prefix) or not base.endswith(suffix):
        return None
    return base[len(prefix) : -len(suffix)]


def _year_timepoints(root: Path) -> dict:
    out: dict[str, list[str]] = {}
    for year_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        tps = sorted(p.name for p in year_dir.iterdir() if p.is_dir())
        if tps:
            out[year_dir.name] = tps
    return out


def _levels_for(root: Path, study_id: str, year: str, timepoint: str) -> list[str]:
    base = root / year / timepoint
    found = []
    for f in glob(str(base / "agg_yield_estimates_*.csv")):
        lvl = _extract_level(f, study_id, year, timepoint)
        if lvl:
            found.append(lvl)
    return sorted(found)


def _raster_path(root: Path, year: str, timepoint: str, kind: str) -> Optional[Path]:
    prefix = RASTER_KINDS[kind][0]
    base = root / year / timepoint
    hits = [p for p in base.glob(f"{prefix}*.tif") if p.name.startswith(prefix)]
    return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------------- #


def _stats_csv(root: Path, study_id: str, year: str, timepoint: str, level: str) -> Path:
    path = root / year / timepoint / f"agg_yield_estimates_{level}_{study_id}_{year}_{timepoint}.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No {level} statistics for {year}/{timepoint}.")
    return path


def _load_stats(path: Path) -> dict:
    df = pd.read_csv(path)
    if "region" not in df.columns:
        raise HTTPException(status_code=500, detail=f"{path.name} has no 'region' column.")
    # The primary-level file carries one row per region of the *whole* source shapefile,
    # so a study covering 22 of Kenya's 290 ADM2s ships 268 all-null rows. Drop rows with
    # no values at all - they have no geometry on the map either.
    metrics = [c for c in df.columns if c != "region"]
    if metrics:
        df = df.dropna(subset=metrics, how="all")
    # Cast to object first: assigning None into a float column just puts NaN back, and
    # NaN is not valid JSON - it makes the whole response fail to serialise.
    df = df.astype(object).where(pd.notnull(df), None)
    return {str(row.pop("region")): row for row in df.to_dict(orient="records")}


# --------------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------------- #


def _geometry_source(root: Path, study_id: str, year: str, timepoint: str, level: str) -> tuple[Path, str]:
    """Return (file, id column) holding this level's polygons.

    The per-region simulation level ("primary") has its boundaries written per
    year/timepoint; the aggregation levels are drawn from the study's aggregation
    shapefiles, named by the run config.
    """
    if level == PRIMARY_LEVEL:
        hits = sorted((root / year / timepoint).glob("aggregated_region_boundaries_*.geojson"))
        if not hits:
            raise HTTPException(status_code=404, detail=f"No primary boundaries for {year}/{timepoint}.")
        return hits[0], "region"

    try:
        cfg = get_run_config(studies_dir, study_id)
        levels = (cfg.get("eval_params") or {}).get("aggregation_levels") or {}
    except Exception:
        levels = {}
    entry = levels.get(level)
    if not entry:
        # Older studies name agg_yield_estimates_* after the aggregation shapefile's stem
        # rather than the config key (ukraine has key "mykolayiv_fields" but files named
        # "Mykolaiv_Field_Scale_Yield"), so fall back to matching on the stem.
        for candidate in levels.values():
            stem = Path(str(candidate.get("shapefile") or "")).stem
            if stem and stem == level:
                entry = candidate
                break
    if not entry:
        raise HTTPException(status_code=404, detail=f"Level '{level}' is not configured for this study.")

    shp = str(entry.get("shapefile") or "")
    candidates = [Path(shp), _study_root(study_id) / "aggregation_shapefiles" / shp]
    if root != _study_root(study_id):
        # Snapshot first, and by basename: package_and_upload copies these into the
        # snapshot's own aggregation_shapefiles/, and a config that names an absolute
        # path would otherwise resolve straight back to the live study's copy.
        candidates.insert(0, root / "aggregation_shapefiles" / Path(shp).name)
    for c in candidates:
        if c.is_file():
            return c, entry.get("name_column") or "region"
    raise HTTPException(status_code=404, detail=f"Aggregation shapefile for '{level}' not found.")


@lru_cache(maxsize=32)
def _read_geometry(path: str, id_column: str, _fp: str) -> gpd.GeoDataFrame:
    """Polygons only, one row per region.

    Aggregation shapefiles carry one row per region *per year* (same geometry repeated
    with that year's reference yield), so deduplicate - the map needs each shape once
    and takes every number from the stats CSV instead.
    """
    gdf = gpd.read_file(path)
    if id_column not in gdf.columns:
        raise HTTPException(status_code=500, detail=f"'{id_column}' missing from {os.path.basename(path)}.")
    gdf = gdf[[id_column, "geometry"]].rename(columns={id_column: "region"})
    gdf["region"] = gdf["region"].astype(str)
    gdf = gdf.drop_duplicates(subset="region", keep="first").reset_index(drop=True)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def _build_geometry(path: Path, id_column: str, zoom: int, bbox: Optional[tuple]) -> dict:
    gdf = _read_geometry(str(path), id_column, _fingerprint(path))

    if bbox:
        minx, miny, maxx, maxy = bbox
        gdf = gdf.cx[minx:maxx, miny:maxy]

    px = _deg_per_pixel(zoom)
    if not gdf.empty:
        # Drop what cannot render: anything whose whole extent is under half a pixel.
        extent = gdf.bounds
        big_enough = ((extent.maxx - extent.minx) > px * DROP_BELOW_PIXELS) | (
            (extent.maxy - extent.miny) > px * DROP_BELOW_PIXELS
        )
        gdf = gdf[big_enough]

    if not gdf.empty:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(px * SIMPLIFY_PIXEL_FRACTION, preserve_topology=True)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

    return json.loads(gdf.to_json())


def _parse_bbox(bbox: Optional[str]) -> Optional[tuple]:
    if not bbox:
        return None
    try:
        parts = [float(x) for x in bbox.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox must be 'minx,miny,maxx,maxy'.")
    if len(parts) != 4:
        raise HTTPException(status_code=400, detail="bbox must have four values.")
    minx, miny, maxx, maxy = parts
    if not (math.isfinite(minx) and math.isfinite(miny) and math.isfinite(maxx) and math.isfinite(maxy)):
        raise HTTPException(status_code=400, detail="bbox values must be finite.")
    if minx > maxx or miny > maxy:
        raise HTTPException(status_code=400, detail="bbox min must not exceed max.")
    return (minx, miny, maxx, maxy)


# --------------------------------------------------------------------------------- #
# Handlers (shared by the live-study and archived-run routes)
# --------------------------------------------------------------------------------- #


def _manifest(study_id: str, run_id: Optional[str]) -> dict:
    root = _resolve_root(study_id, run_id)
    yt = _year_timepoints(root)
    if not yt:
        raise HTTPException(status_code=404, detail="No results available yet.")

    first_year = sorted(yt)[-1]
    first_tp = yt[first_year][0]
    levels = _levels_for(root, study_id, first_year, first_tp)

    all_timepoints = sorted({tp for tps in yt.values() for tp in tps})

    rasters = []
    for kind, (_prefix, label) in RASTER_KINDS.items():
        if _raster_path(root, first_year, first_tp, kind):
            rasters.append({"kind": kind, "label": label})

    return {
        "study_id": study_id,
        "run_id": run_id,
        "cropmasks": _cropmasks_by_year(study_id),
        # Study extent, so the map can frame the region immediately on load instead of
        # waiting for a geometry response - which never arrives if the default world view
        # happens not to overlap the study.
        "bounds": _study_bounds(root, study_id, first_year, first_tp, levels),
        # Observed-LAI window per year/timepoint, straight from lai_params.time_bounds.
        # The map turns this into the month range for Sentinel-2 imagery, so the
        # imagery you can browse is exactly the period the yield was derived from.
        "seasons": _seasons(study_id),
        "years": yt,
        "levels": levels,
        "primary_level": PRIMARY_LEVEL,
        "rasters": rasters,
        # Per-region report plots are keyed by primary region id, so only that level
        # offers them; the frontend hides the plot pane on aggregation levels.
        "plots_level": PRIMARY_LEVEL,
        # Default colour domains, shared across years and levels. The frontend uses these
        # unless the user pins their own min/max in the legend.
        "scales": _scales(root, study_id, levels, all_timepoints),
        # Which levels carry which values, so the toolbar can disable a choice and say
        # where it is available instead of dropping it from the list.
        "metrics_by_level": _metrics_by_level(root, study_id, levels, all_timepoints),
        "default": {"year": first_year, "timepoint": first_tp, "level": levels[0] if levels else PRIMARY_LEVEL},
    }


# Metrics the map can colour by. A shared domain is computed once per study across every
# level and every year so a colour means the same value wherever you are - switching year
# or admin level no longer silently rescales the legend under you. Diverging metrics get a
# symmetric domain about zero.
SCALE_METRICS = [
    "mean_yield_kg_ha",
    "median_yield_kg_ha",
    "mean_yield_kg_ha_apsim",
    "reported_mean_yield_kg_ha",
    "total_production_ton",
    "total_area_ha",
    "max_rs_lai",
    "abs_error",
    "rel_error",
]
DIVERGING_SCALE_METRICS = {"rel_error"}


def _scales(root: Path, study_id: str, levels: list, timepoints: list) -> dict:
    """{metric: {min, max}} over every level/year, from the pipeline's all_predictions files.

    Capped at the 95th percentile rather than the maximum, for the same reason the
    per-year domain was: one outlying region otherwise pushes every other value into the
    bottom of the ramp.
    """
    frames = []
    for level in list(levels) + [PRIMARY_LEVEL]:
        for tp in timepoints:
            path = _all_predictions_path(root, study_id, level, tp)
            if path is not None:
                try:
                    frames.append(pd.read_csv(path))
                except Exception:
                    continue
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)

    pred = pd.to_numeric(df.get("mean_yield_kg_ha"), errors="coerce")
    ref = pd.to_numeric(df.get("reported_mean_yield_kg_ha"), errors="coerce")
    if pred is not None and ref is not None:
        df["abs_error"] = (pred - ref).abs()
        df["rel_error"] = ((pred - ref) / ref.replace(0, float("nan"))) * 100

    out: dict = {}
    for metric in SCALE_METRICS:
        if metric not in df.columns:
            continue
        v = pd.to_numeric(df[metric], errors="coerce").dropna()
        v = v[v.apply(math.isfinite)]
        if v.empty:
            continue
        if metric in DIVERGING_SCALE_METRICS:
            bound = float(v.abs().quantile(0.95))
            if bound > 0:
                out[metric] = {"min": -bound, "max": bound}
        else:
            lo, hi = float(v.min()), float(v.quantile(0.95))
            out[metric] = {"min": lo, "max": hi if hi > lo else float(v.max())}
    return out


def _metrics_by_level(root: Path, study_id: str, levels: list, timepoints: list) -> dict:
    """{level: [metric keys that level actually has values for]}.

    The toolbar shows the same "colour by" list at every level and disables what is not
    there, rather than silently shortening the list - a list whose entries appear and
    disappear as you change level reads as a bug, and gives no hint that the value exists
    somewhere else. Derived from the concatenated predictions files, so it costs one small
    read per level rather than a scan of every year.
    """
    out: dict[str, list] = {}
    for level in list(levels) + [PRIMARY_LEVEL]:
        cols: set[str] = set()
        for tp in timepoints:
            path = _all_predictions_path(root, study_id, level, tp)
            if path is None:
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            for metric in SCALE_METRICS:
                if metric in df.columns and pd.to_numeric(df[metric], errors="coerce").notna().any():
                    cols.add(metric)
            # Error metrics are derived, so they exist wherever both sides do.
            if {"mean_yield_kg_ha", "reported_mean_yield_kg_ha"} <= cols:
                cols.update({"abs_error", "rel_error"})
        if cols:
            out[level] = sorted(cols)
    return out


def _study_bounds(root: Path, study_id: str, year: str, timepoint: str, levels: list) -> Optional[list]:
    """[west, south, east, north] in WGS84 covering the study's regions, or None."""
    for level in [PRIMARY_LEVEL] + list(levels):
        try:
            path, id_col = _geometry_source(root, study_id, year, timepoint, level)
            gdf = _read_geometry(str(path), id_col, _fingerprint(path))
        except Exception:
            continue
        if gdf.empty:
            continue
        b = gdf.total_bounds
        if all(math.isfinite(v) for v in b):
            return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    return None


def _seasons(study_id: str) -> dict:
    """{year: {timepoint: [start, end]}} from the run config's LAI time bounds."""
    try:
        cfg = get_run_config(studies_dir, study_id)
        bounds = (cfg.get("lai_params") or {}).get("time_bounds") or {}
    except Exception:
        return {}
    out: dict = {}
    for year, tps in bounds.items():
        if not isinstance(tps, dict):
            continue
        for tp, window in tps.items():
            if isinstance(window, (list, tuple)) and len(window) == 2 and all(window):
                out.setdefault(str(year), {})[str(tp)] = [str(window[0]), str(window[1])]
    return out


def _cropmasks_by_year(study_id: str) -> dict:
    """year -> cropmask name, from the run config's lai_params.crop_mask.

    Names only: the frontend draws them through /api/cropmasks/{name}/tiles, so there is
    one cropmask tiler rather than a second copy of it here.
    """
    try:
        cfg = get_run_config(studies_dir, study_id)
        masks = (cfg.get("lai_params") or {}).get("crop_mask") or {}
    except Exception:
        return {}
    out = {}
    for year, path in masks.items():
        if not path:
            continue
        name = Path(str(path)).name
        out[str(year)] = name[: -len(".tif")] if name.endswith(".tif") else name
    return out


# Columns of evaluation_{level}.csv worth surfacing, with their display labels. These are
# the pipeline's own numbers (vercye_ops/evaluation) - the map reports them rather than
# recomputing anything, so what the UI shows and what the reports show cannot drift.
EVAL_FIELDS: list[tuple[str, str, int]] = [
    ("n_regions", "Regions evaluated", 0),
    ("r2_scikit", "R\u00b2", 3),
    ("rmse_kg_ha", "RMSE (kg/ha)", 1),
    ("rrmse", "Relative RMSE (%)", 1),
    ("mape", "MAPE", 3),
    ("mean_err_kg_ha", "Mean error (kg/ha)", 1),
    ("median_err_kg_ha", "Median error (kg/ha)", 1),
    ("mean_abs_err_kg_ha", "Mean absolute error (kg/ha)", 1),
    ("median_abs_err_kg_ha", "Median absolute error (kg/ha)", 1),
]


def _read_evaluation(root: Path, year: str, timepoint: str, level: str) -> Optional[dict]:
    """One row of pipeline-computed accuracy metrics, or None if the level has no
    reference data for that year (in which case the pipeline writes no evaluation file)."""
    path = root / year / timepoint / f"evaluation_{level}.csv"
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    row = df.iloc[0]
    out: dict = {}
    for key, _label, _digits in EVAL_FIELDS:
        if key in row and pd.notnull(row[key]):
            out[key] = float(row[key])
    return out or None


def _evaluation(study_id: str, run_id: Optional[str], year: int, timepoint: str, level: str) -> dict:
    root = _resolve_root(study_id, run_id)
    metrics = _read_evaluation(root, str(year), timepoint, level)
    return {
        "level": level,
        "year": str(year),
        "timepoint": timepoint,
        "fields": [{"key": k, "label": lab, "digits": d} for k, lab, d in EVAL_FIELDS],
        "metrics": metrics,
    }


def _all_predictions_path(root: Path, study_id: str, level: str, timepoint: str) -> Optional[Path]:
    path = root / f"all_predictions_{study_id}_{level}_{timepoint}.csv"
    return path if path.is_file() else None


def _multiyear(study_id: str, run_id: Optional[str], timepoint: str, level: str) -> dict:
    """Every year of a level in one payload, for the summary panel's multiyear view.

    Both halves come straight off disk: the per-year accuracy metrics out of the
    pipeline's ``evaluation_{level}.csv``, and the per-region predicted/reported yields
    out of ``all_predictions_{study}_{level}_{tp}.csv``, which the pipeline writes by
    concatenating the same ``agg_yield_estimates`` files the single-year view reads.
    Nothing is recomputed here - a number shown in this panel is a number the pipeline
    produced.
    """
    root = _resolve_root(study_id, run_id)
    years = [y for y, tps in _year_timepoints(root).items() if timepoint in tps]
    if not years:
        raise HTTPException(status_code=404, detail=f"No results for timepoint '{timepoint}'.")

    metrics = []
    for year in sorted(years):
        row = _read_evaluation(root, year, timepoint, level)
        if row:
            metrics.append({"year": year, **row})

    regions: dict[str, list] = {}
    path = _all_predictions_path(root, study_id, level, timepoint)
    if path is not None:
        df = pd.read_csv(path)
        want = [c for c in ("year", "region", "mean_yield_kg_ha", "reported_mean_yield_kg_ha") if c in df.columns]
        if "year" in want and "region" in want:
            df = df[want].astype(object).where(pd.notnull(df[want]), None)
            for rec in df.to_dict(orient="records"):
                regions.setdefault(str(rec["region"]), []).append(
                    {
                        "year": str(rec["year"]),
                        "predicted": rec.get("mean_yield_kg_ha"),
                        "reported": rec.get("reported_mean_yield_kg_ha"),
                    }
                )
            for series in regions.values():
                series.sort(key=lambda r: r["year"])

    return {
        "level": level,
        "timepoint": timepoint,
        "years": sorted(years),
        "fields": [{"key": k, "label": lab, "digits": d} for k, lab, d in EVAL_FIELDS],
        "metrics": metrics,
        "source": path.name if path is not None else None,
        "regions": regions,
    }


def _stats(study_id: str, run_id: Optional[str], year: int, timepoint: str, level: str) -> dict:
    root = _resolve_root(study_id, run_id)
    path = _stats_csv(root, study_id, str(year), timepoint, level)
    values = _load_stats(path)
    numeric = [v.get("mean_yield_kg_ha") for v in values.values() if isinstance(v.get("mean_yield_kg_ha"), (int, float))]
    return {
        "level": level,
        "year": str(year),
        "timepoint": timepoint,
        "source": path.name,
        "range": {"min": min(numeric), "max": max(numeric)} if numeric else None,
        "regions": values,
    }


def _lai(study_id: str, run_id: Optional[str], year: int, timepoint: str, level: str) -> dict:
    """Observed-LAI curves per region, from agg_lai_timeseries_{level}_*.csv.

    Served as parallel arrays (dates once, one value array per region) rather than a row
    per point: the ADM2 file for a 22-region study is already 732 rows, and a national
    study's is far larger, so the row-oriented form would dominate the payload.
    """
    root = _resolve_root(study_id, run_id)
    base = root / str(year) / timepoint
    path = base / f"agg_lai_timeseries_{level}_{study_id}_{year}_{timepoint}.csv"

    if path.is_file():
        df = pd.read_csv(path)
        if "Date" not in df.columns or "region" not in df.columns:
            raise HTTPException(status_code=500, detail=f"{path.name} is missing Date/region columns.")
    elif level == PRIMARY_LEVEL:
        # The pipeline only aggregates LAI for the *configured* aggregation levels - the
        # rule's wildcard is constrained to them and excludes "primary" outright - so no
        # study has agg_lai_timeseries_primary_*. The same series exists per region as
        # <region>/<region>_LAI_STATS.csv, so assemble it from those rather than asking
        # for a pipeline change.
        frames = []
        for stats_file in sorted(base.glob("*/*_LAI_STATS.csv")):
            region = stats_file.parent.name
            one = pd.read_csv(stats_file)
            if "Date" not in one.columns:
                continue
            one["region"] = region
            frames.append(one)
        if not frames:
            raise HTTPException(status_code=404, detail=f"No LAI timeseries for {level} {year}/{timepoint}.")
        df = pd.concat(frames, ignore_index=True)
    else:
        raise HTTPException(status_code=404, detail=f"No LAI timeseries for {level} {year}/{timepoint}.")

    # Dates are written day-first (01/02/2024 is 1 February); parsing them the other way
    # silently reorders the season.
    df["_d"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["_d"]).sort_values("_d")
    dates = [d.strftime("%Y-%m-%d") for d in sorted(df["_d"].unique())]
    index = {d: i for i, d in enumerate(dates)}

    # Crop-adjusted median is what the pipeline matches against, so it is the default.
    series_cols = [c for c in ("LAI Median Adjusted", "LAI Mean Adjusted", "LAI Median", "LAI Mean") if c in df.columns]
    if not series_cols:
        raise HTTPException(status_code=500, detail=f"{path.name} has no LAI value column.")
    col = series_cols[0]

    regions: dict[str, list] = {}
    for region, grp in df.groupby("region"):
        values: list = [None] * len(dates)
        for _, row in grp.iterrows():
            v = row[col]
            values[index[row["_d"].strftime("%Y-%m-%d")]] = None if pd.isna(v) else float(v)
        regions[str(region)] = values

    return {"level": level, "column": col, "dates": dates, "regions": regions}


def _geometry(
    study_id: str, run_id: Optional[str], year: int, timepoint: str, level: str, zoom: int, bbox: Optional[str]
) -> Response:
    root = _resolve_root(study_id, run_id)
    path, id_col = _geometry_source(root, study_id, str(year), timepoint, level)
    etag = f'W/"{_content_etag(path)}-{level}-z{zoom}-{bbox or "full"}"'
    payload = _build_geometry(path, id_col, zoom, _parse_bbox(bbox))
    return Response(
        content=json.dumps(payload),
        media_type="application/geo+json",
        headers={"ETag": etag, "Cache-Control": "private, max-age=300"},
    )


def _tile(
    study_id: str, run_id: Optional[str], year: int, timepoint: str, kind: str, z: int, x: int, y: int, rescale: str
) -> Response:
    if kind not in RASTER_KINDS:
        raise HTTPException(status_code=404, detail=f"Unknown raster '{kind}'.")
    root = _resolve_root(study_id, run_id)
    # No fallback to the live study: snapshots carry their own EPSG:4326 mosaics, and
    # borrowing the live one would paint the *current* run's pixels under an archived
    # run's label. A snapshot taken before those mosaics were packaged simply has no
    # pixel layer - the manifest already omits it, so the frontend never offers it.
    path = _raster_path(root, str(year), timepoint, kind)
    if path is None:
        raise HTTPException(status_code=404, detail=f"No {kind} raster for {year}/{timepoint}.")

    try:
        lo, hi = (float(v) for v in rescale.split(","))
    except ValueError:
        raise HTTPException(status_code=400, detail="rescale must be 'min,max'.")

    from rio_tiler.errors import TileOutsideBounds
    from rio_tiler.io import Reader

    try:
        with Reader(str(path)) as src:
            img = src.tile(x, y, z, tilesize=256)
    except TileOutsideBounds:
        # Transparent 256x256 rather than a 404, so Leaflet does not log errors while
        # panning past the study's extent.
        return Response(content=_blank_tile(), media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})

    img.rescale(in_range=((lo, hi),))
    png = img.render(img_format="PNG", colormap=_colormap())
    # Tiles for a finished run never change.
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@lru_cache(maxsize=1)
def _blank_tile() -> bytes:
    import numpy as np
    from rio_tiler.models import ImageData

    # Fully masked -> fully transparent PNG.
    arr = np.ma.MaskedArray(
        np.zeros((1, 256, 256), dtype="uint8"),
        mask=np.ones((1, 256, 256), dtype=bool),
    )
    return ImageData(arr).render(img_format="PNG")


@lru_cache(maxsize=1)
def _colormap() -> dict:
    """Viridis, matching `rampColor` in components/ResultsMap/scale.ts.

    Viridis is what the original prebuilt map used and it is the right choice here: it is
    perceptually uniform and monotonic in lightness, so equal steps in yield look like
    equal steps on screen, and it separates neighbouring values far better than the
    single-hue green it briefly replaced. The two implementations must stay identical or a
    region's fill and the pixels inside it read as different values. Index 0 is fully
    transparent so no-data and true zero-yield cropland do not paint over the basemap."""
    seq = [
        (68, 1, 84),
        (72, 40, 120),
        (62, 74, 137),
        (49, 104, 142),
        (38, 130, 142),
        (31, 158, 137),
        (53, 183, 121),
        (110, 206, 88),
        (181, 222, 43),
        (253, 231, 37),
    ]
    cmap = {}
    for i in range(256):
        x = (i / 255.0) * (len(seq) - 1)
        lo = min(len(seq) - 2, int(x))
        f = x - lo
        r, g, b = (round(seq[lo][c] + (seq[lo + 1][c] - seq[lo][c]) * f) for c in range(3))
        cmap[i] = (r, g, b, 0 if i == 0 else 220)
    return cmap


def _raster_stats(study_id: str, run_id: Optional[str], year: int, timepoint: str, kind: str) -> dict:
    """Percentiles for the mosaic, so the legend and the tile rescale reflect the actual
    pixel distribution instead of a guessed range. Read from the COG's overviews, so this
    costs ~0.1 s even on a 46k x 46k raster."""
    if kind not in RASTER_KINDS:
        raise HTTPException(status_code=404, detail=f"Unknown raster '{kind}'.")
    root = _resolve_root(study_id, run_id)
    path = _raster_path(root, str(year), timepoint, kind)
    if path is None:
        raise HTTPException(status_code=404, detail=f"No {kind} raster for {year}/{timepoint}.")
    return _raster_stats_cached(str(path), _fingerprint(path))


@lru_cache(maxsize=32)
def _raster_stats_cached(path: str, _fp: str) -> dict:
    from rio_tiler.io import Reader

    with Reader(path) as src:
        st = src.statistics()["b1"]
    return {
        "min": float(st.min),
        "max": float(st.max),
        "p2": float(st.percentile_2),
        "p98": float(st.percentile_98),
    }


def _region_plot(study_id: str, run_id: Optional[str], year: int, timepoint: str, region: str) -> Response:
    """One region's report PNG, read directly out of the map zip.

    The legacy endpoint extracts the whole archive on every request (189 MiB for
    Ukraine); this reads a single member and streams it.
    """
    # Plots exist only for the per-region simulation level: the zip members are named
    # "{primary_region_id}_yield_report.png", so an aggregation-level name like "Bumula"
    # has no plot. The manifest advertises this via "plots_level".
    if "/" in region or "\\" in region or ".." in region:
        raise HTTPException(status_code=400, detail="Invalid region id.")
    root = _resolve_root(study_id, run_id)
    zips = sorted((root / str(year) / timepoint).glob("interactive_map_*.zip"))
    if not zips:
        raise HTTPException(status_code=404, detail=f"No region plots for {year}/{timepoint}.")
    member = f"{region}_yield_report.png"
    try:
        with zipfile.ZipFile(zips[0]) as z:
            data = z.read(member)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No plot for region '{region}'.")
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


# --------------------------------------------------------------------------------- #
# Routes - live study
# --------------------------------------------------------------------------------- #


@router.get("/{study_id}/results/manifest")
def results_manifest(study_id: StudyID):
    return _manifest(study_id, None)


@router.get("/{study_id}/results/{year}/{timepoint}/levels/{level}/geometry")
def results_geometry(
    study_id: StudyID,
    year: int,
    timepoint: str,
    level: str,
    zoom: int = Query(6, ge=0, le=22),
    bbox: Optional[str] = Query(None),
):
    return _geometry(study_id, None, year, timepoint, level, zoom, bbox)


@router.get("/{study_id}/results/{year}/{timepoint}/levels/{level}/stats")
def results_stats(study_id: StudyID, year: int, timepoint: str, level: str):
    return _stats(study_id, None, year, timepoint, level)


@router.get("/{study_id}/results/{year}/{timepoint}/tiles/{kind}/{z}/{x}/{y}.png")
def results_tile(
    study_id: StudyID,
    year: int,
    timepoint: str,
    kind: str,
    z: int,
    x: int,
    y: int,
    rescale: str = Query("0,5000"),
):
    return _tile(study_id, None, year, timepoint, kind, z, x, y, rescale)


@router.get("/{study_id}/results/{year}/{timepoint}/levels/{level}/lai")
def results_lai(study_id: StudyID, year: int, timepoint: str, level: str):
    return _lai(study_id, None, year, timepoint, level)


@router.get("/{study_id}/results/{year}/{timepoint}/levels/{level}/evaluation")
def results_evaluation(study_id: StudyID, year: int, timepoint: str, level: str):
    return _evaluation(study_id, None, year, timepoint, level)


@router.get("/{study_id}/results/multiyear/{timepoint}/levels/{level}")
def results_multiyear(study_id: StudyID, timepoint: str, level: str):
    return _multiyear(study_id, None, timepoint, level)


@router.get("/{study_id}/results/{year}/{timepoint}/rasters/{kind}/stats")
def results_raster_stats(study_id: StudyID, year: int, timepoint: str, kind: str):
    return _raster_stats(study_id, None, year, timepoint, kind)


@router.get("/{study_id}/results/{year}/{timepoint}/regions/{region}/plot.png")
def results_region_plot(study_id: StudyID, year: int, timepoint: str, region: str):
    return _region_plot(study_id, None, year, timepoint, region)


# --------------------------------------------------------------------------------- #
# Routes - archived run
# --------------------------------------------------------------------------------- #


@router.get("/{study_id}/runs/{run_id}/results/manifest")
def run_results_manifest(study_id: StudyID, run_id: RunID):
    return _manifest(study_id, run_id)


@router.get("/{study_id}/runs/{run_id}/results/{year}/{timepoint}/levels/{level}/geometry")
def run_results_geometry(
    study_id: StudyID,
    run_id: RunID,
    year: int,
    timepoint: str,
    level: str,
    zoom: int = Query(6, ge=0, le=22),
    bbox: Optional[str] = Query(None),
):
    return _geometry(study_id, run_id, year, timepoint, level, zoom, bbox)


@router.get("/{study_id}/runs/{run_id}/results/{year}/{timepoint}/levels/{level}/stats")
def run_results_stats(study_id: StudyID, run_id: RunID, year: int, timepoint: str, level: str):
    return _stats(study_id, run_id, year, timepoint, level)


@router.get("/{study_id}/runs/{run_id}/results/{year}/{timepoint}/tiles/{kind}/{z}/{x}/{y}.png")
def run_results_tile(
    study_id: StudyID,
    run_id: RunID,
    year: int,
    timepoint: str,
    kind: str,
    z: int,
    x: int,
    y: int,
    rescale: str = Query("0,5000"),
):
    return _tile(study_id, run_id, year, timepoint, kind, z, x, y, rescale)


@router.get("/{study_id}/runs/{run_id}/results/{year}/{timepoint}/levels/{level}/lai")
def run_results_lai(study_id: StudyID, run_id: RunID, year: int, timepoint: str, level: str):
    return _lai(study_id, run_id, year, timepoint, level)


@router.get("/{study_id}/runs/{run_id}/results/{year}/{timepoint}/levels/{level}/evaluation")
def run_results_evaluation(study_id: StudyID, run_id: RunID, year: int, timepoint: str, level: str):
    return _evaluation(study_id, run_id, year, timepoint, level)


@router.get("/{study_id}/runs/{run_id}/results/multiyear/{timepoint}/levels/{level}")
def run_results_multiyear(study_id: StudyID, run_id: RunID, timepoint: str, level: str):
    return _multiyear(study_id, run_id, timepoint, level)


@router.get("/{study_id}/runs/{run_id}/results/{year}/{timepoint}/rasters/{kind}/stats")
def run_results_raster_stats(study_id: StudyID, run_id: RunID, year: int, timepoint: str, kind: str):
    return _raster_stats(study_id, run_id, year, timepoint, kind)


@router.get("/{study_id}/runs/{run_id}/results/{year}/{timepoint}/regions/{region}/plot.png")
def run_results_region_plot(study_id: StudyID, run_id: RunID, year: int, timepoint: str, region: str):
    return _region_plot(study_id, run_id, year, timepoint, region)

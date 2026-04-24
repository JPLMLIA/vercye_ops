"""One-shot migration: convert an existing standardized-lai directory of
Float32 LAI tiles to the new Int16 storage format and regenerate the
merged-lai daily VRTs so downstream readers still see Float32 + NaN.

Atomic per-file: writes <name>.int16.tmp, verifies it re-opens and decodes to
the same Float32 values (within the 0.001 quantum), then deletes the original
and renames. Safe to re-run after interruption — files already Int16 are
skipped.

Usage (from anywhere with the vercye_ops package importable):
    python migrate_standardized_to_int16.py \
        --standardized-dir /data/vercye/lai/kenya/standardized-lai \
        --merged-dir       /data/vercye/lai/kenya/merged-lai \
        --geojson-path     /data/vercye/lai/kenya/region.geojson \
        --resolution       10 \
        --region-out-prefix kenya \
        --num-workers      32
"""

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import click
import numpy as np
import rasterio as rio

from vercye_ops.lai.lai_creation_STAC.vrt_patcher import patch_vrt_for_int16_sources
from vercye_ops.utils.init_logger import get_logger

LAI_SCALE = 0.001
LAI_NODATA = -32768

logger = get_logger()
logger.setLevel("INFO")


def is_already_int16(path):
    try:
        with rio.open(path) as s:
            return s.dtypes[0] == "int16"
    except Exception:
        return False


def convert_one(path):
    """Convert one Float32 standardized tile to Int16 in place (atomic).

    Returns (path, status, detail).
    """
    path = str(path)
    try:
        if is_already_int16(path):
            return path, "skipped_already_int16", None

        tmp = path + ".int16.tmp"
        if os.path.exists(tmp):
            os.remove(tmp)

        with rio.open(path) as src:
            if src.count != 1:
                return path, "failed", f"expected 1 band, got {src.count}"
            meta = src.meta.copy()
            data_f = src.read(1).astype(np.float32, copy=False)

            # Quantize: negatives (and NaN) -> LAI_NODATA; positives -> round/LAI_SCALE
            out = np.full(data_f.shape, LAI_NODATA, dtype=np.int16)
            valid = np.isfinite(data_f) & (data_f >= 0)
            if valid.any():
                scaled = np.round(data_f[valid] / LAI_SCALE)
                np.clip(scaled, 0, 32767, out=scaled)
                out[valid] = scaled.astype(np.int16)

            meta.update(
                {
                    "dtype": "int16",
                    "nodata": LAI_NODATA,
                    "compress": "zstd",
                    "predictor": 2,
                    "zstd_level": 19,
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                }
            )

            with rio.open(tmp, "w", **meta) as dst:
                dst.write(out, 1)
                dst.scales = (LAI_SCALE,)
                dst.offsets = (0.0,)

        # Verify: reopen and check max abs diff is within half a quantum
        with rio.open(path) as src, rio.open(tmp) as vsrc:
            orig = src.read(1)
            vnew = vsrc.read(1).astype(np.float32) * LAI_SCALE
            vnew[vsrc.read(1) == LAI_NODATA] = np.nan
            # Expected difference: NaN positions either align (orig NaN/neg)
            # or orig was a negative float that we intentionally mapped to nodata.
            orig_invalid = ~np.isfinite(orig) | (orig < 0)
            vnew_invalid = np.isnan(vnew)
            if not np.array_equal(orig_invalid, vnew_invalid):
                os.remove(tmp)
                n_diff = int((orig_invalid != vnew_invalid).sum())
                return path, "failed", f"invalid-mask mismatch on {n_diff} pixels"
            finite = ~orig_invalid
            if finite.any():
                max_diff = float(np.abs(orig[finite] - vnew[finite]).max())
                # Roundtrip bound: half quantum (0.0005) plus float32 decode slop.
                # Float32 representation of LAI_SCALE is not exact, so reconstruction
                # adds up to ~1e-6 times the max magnitude (~8 for LAI) on top.
                tolerance = LAI_SCALE / 2 + 1e-5
                if max_diff > tolerance:
                    os.remove(tmp)
                    return path, "failed", f"max diff {max_diff} exceeds tolerance {tolerance}"

        # Atomic replace
        os.replace(tmp, path)
        return path, "converted", None

    except Exception as e:
        # Clean up any partial temp file
        try:
            if os.path.exists(path + ".int16.tmp"):
                os.remove(path + ".int16.tmp")
        except OSError:
            pass
        return path, "failed", f"{type(e).__name__}: {e}"


def rebuild_merged_vrts(standardized_dir, merged_dir, geojson_path, resolution, region_out_prefix):
    """Rebuild all merged-lai daily VRTs from scratch. Assumes the standardized
    tiles are now Int16. Calls gdalbuildvrt then patches the resulting VRT."""
    import geopandas as gpd
    from collections import defaultdict

    standardized_dir = Path(standardized_dir)
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    lai_files = sorted(glob(str(standardized_dir / f"*_{resolution}m_*_LAI_tile_standardized.tif")))
    if not lai_files:
        logger.warning(f"No standardized LAI files found in {standardized_dir}")
        return

    # Read one sample to pull CRS + resolution
    with rio.open(lai_files[0]) as src:
        crs = src.crs
        crs_str = crs.to_string()
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)

    gdf = gpd.read_file(geojson_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    minx, miny, maxx, maxy = gdf.total_bounds

    groups = defaultdict(list)
    for f in lai_files:
        date = Path(f).stem.split("_")[-4]
        groups[date].append(f)

    logger.info(f"Rebuilding {len(groups)} daily VRTs in {merged_dir}")
    for date, paths in sorted(groups.items()):
        out_file = merged_dir / f"{region_out_prefix}_{resolution}m_{date}_LAI.vrt"
        cmd = [
            "gdalbuildvrt",
            "-overwrite",
            "-tap",
            "-te", str(minx), str(miny), str(maxx), str(maxy),
            "-tr", str(res_x), str(res_y),
            "-a_srs", crs_str,
            str(out_file),
        ] + paths
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            logger.error(f"gdalbuildvrt failed for {date}: {res.stderr.decode()}")
            continue
        patch_vrt_for_int16_sources(out_file)
        logger.info(f"  built {out_file.name}  ({len(paths)} tiles)")


@click.command()
@click.option("--standardized-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--merged-dir", required=True, type=click.Path(file_okay=False))
@click.option("--geojson-path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--resolution", required=True, type=int)
@click.option("--region-out-prefix", required=True, type=str)
@click.option("--num-workers", type=int, default=32)
@click.option("--dry-run", is_flag=True, help="List what would be converted, don't modify.")
@click.option("--skip-vrt-rebuild", is_flag=True, help="Convert tiles only, don't rebuild VRTs.")
@click.option("--limit", type=int, default=None, help="Process at most N files (for testing).")
def main(standardized_dir, merged_dir, geojson_path, resolution, region_out_prefix,
         num_workers, dry_run, skip_vrt_rebuild, limit):
    files = sorted(glob(os.path.join(standardized_dir, "*_LAI_tile_standardized.tif")))
    if limit:
        files = files[:limit]
    logger.info(f"Found {len(files)} standardized tiles under {standardized_dir}")

    if dry_run:
        by_status = {"already_int16": 0, "needs_convert": 0}
        for f in files:
            by_status["already_int16" if is_already_int16(f) else "needs_convert"] += 1
        logger.info(f"Dry run: {by_status}")
        return

    converted = skipped = failed = 0
    failures = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(convert_one, f) for f in files]
        for i, fut in enumerate(as_completed(futures), 1):
            path, status, detail = fut.result()
            if status == "converted":
                converted += 1
            elif status == "skipped_already_int16":
                skipped += 1
            else:
                failed += 1
                failures.append((path, detail))
            if i % 200 == 0 or i == len(files):
                logger.info(f"  progress: {i}/{len(files)}  converted={converted}  skipped={skipped}  failed={failed}")

    logger.info(f"Migration complete: converted={converted} skipped={skipped} failed={failed}")
    if failures:
        logger.error(f"{len(failures)} failures (first 10):")
        for p, d in failures[:10]:
            logger.error(f"  {p}: {d}")
        if failed > 0:
            sys.exit(2)

    if not skip_vrt_rebuild:
        rebuild_merged_vrts(standardized_dir, merged_dir, geojson_path, resolution, region_out_prefix)


if __name__ == "__main__":
    main()

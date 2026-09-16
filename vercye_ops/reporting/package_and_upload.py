#!/usr/bin/env python3
"""Snapshot VERCYe study outputs locally and optionally upload them to an rclone remote.

Walks the study directory, picks every file whose basename matches a glob in
``output_data_patterns.txt`` and copies them into

    <study_dir>/run_results/<YYYYMMDD_HHMMSS>/

preserving the relative folder structure so that the snapshot looks like a
miniature copy of the study results at the moment the pipeline completed.
This snapshot becomes the source of truth for past runs: when the user
re-prepares the study with tweaked parameters, the existing snapshots stay
intact and can be browsed from the frontend independently of the live
pipeline outputs.

If ``--rclone-target`` is set, the same files are additionally zipped and
uploaded with ``rclone`` to ``<rclone_target>/<rclone_folder_prefix>/<run_id>/``.
The remote folder name carries a ``_N`` suffix if a folder with the same base
name already exists, so re-runs on the same day never overwrite an existing
upload.

``rclone_target`` must be (or start with) a configured rclone remote, e.g.
``gdrive:`` or ``gdrive:vercye_kenya``. Configure remotes once with
``rclone config``.
"""

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

# Directory (relative to the study root) where per-run snapshots live.
RUN_RESULTS_DIRNAME = "run_results"
RUN_META_FILENAME = "_run_meta.json"
APSIM_MAPPING_FILENAME = "_apsim_mapping.json"
APSIM_SUBDIR = "apsim"
# Trailing tag in output_data_patterns.txt marking a pattern for the core bundle too.
CORE_TAG = "@core"


def load_patterns(patterns_file: Path) -> tuple[list[str], list[str]]:
    """Return (all patterns, core patterns).

    A line may carry a trailing ``@core`` tag, marking it as also belonging to the
    small "core data bundle" - the CSVs, reports and multiyear report a collaborator
    needs without the multi-GB rasters. Core is always a subset of the full set, so
    the tag annotates the existing line rather than duplicating it into a second
    file; a patterns file with no tags simply yields no core bundle.
    """
    patterns: list[str] = []
    core: list[str] = []
    for line in patterns_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        is_core = line.endswith(CORE_TAG)
        if is_core:
            line = line[: -len(CORE_TAG)].strip()
            if not line:
                continue
        patterns.append(line)
        if is_core:
            core.append(line)
    if not patterns:
        raise ValueError(f"No patterns loaded from {patterns_file}")
    return patterns, core


def iter_matching_files(root: Path, patterns: list[str]):
    """Yield (absolute path, path relative to root) for files matching ``patterns``.

    The ``run_results`` directory itself is skipped so previous snapshots are
    never re-snapshotted into the new one.
    """
    skip_dirs = {RUN_RESULTS_DIRNAME}
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip the run_results tree in-place so os.walk does not descend.
        dirnames[:] = [d for d in dirnames if d not in skip_dirs or Path(dirpath) != root]
        for name in filenames:
            if any(fnmatch.fnmatch(name, p) for p in patterns):
                full = Path(dirpath) / name
                yield full, full.relative_to(root)


def copy_matches_to_snapshot(root: Path, patterns: list[str], dest_root: Path) -> int:
    count = 0
    for src, rel in iter_matching_files(root, patterns):
        dst = dest_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    return count


def snapshot_apsim_sources(study_root: Path, snapshot_root: Path) -> dict:
    """Persist the APSIM source templates + a region->template mapping.

    The per-region customized ``<region>.apsimx`` files are far too large
    to snapshot (650KB each across thousands of region/year/timepoint
    combinations) — but the *source* templates kept under
    ``<study>/apsim/`` and referenced by ``APSIM_TEMPLATE_PATHS`` in
    ``setup_config.yaml`` are tiny and capture which APSIM file was used
    for which admin region. We copy those plus the mapping so a future
    user can reproduce the run.
    """
    # study_root is sim_study_head_dir (<studies_dir>/<study_id>/<study_id>);
    # setup_config and the apsim/ source dir live one level above.
    setup_cfg_path = study_root.parent / "setup_config.yaml"
    if not setup_cfg_path.exists():
        return {"region_to_template": {}, "files": [], "filter_column": None}

    try:
        import yaml as _yaml

        cfg = _yaml.safe_load(setup_cfg_path.read_text()) or {}
    except Exception:
        cfg = {}

    template_paths = cfg.get("APSIM_TEMPLATE_PATHS") or {}
    filter_col = cfg.get("APSIM_TEMPLATE_PATHS_FILTER_COL_NAME")

    apsim_dest = snapshot_root / APSIM_SUBDIR
    apsim_dest.mkdir(parents=True, exist_ok=True)

    files: dict[str, list[str]] = {}
    region_map: dict[str, str] = {}
    for region, src_path in template_paths.items():
        if not src_path:
            continue
        src = Path(src_path)
        if not src.is_file():
            continue
        dst = apsim_dest / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        region_map[region] = src.name
        files.setdefault(src.name, []).append(region)

    manifest = {
        "filter_column": filter_col,
        "region_to_template": region_map,
        "files": [
            {
                "name": name,
                "size": (apsim_dest / name).stat().st_size if (apsim_dest / name).exists() else None,
                "regions": sorted(regions),
            }
            for name, regions in sorted(files.items())
        ],
    }
    (snapshot_root / APSIM_MAPPING_FILENAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def snapshot_aggregation_shapefiles(study_root: Path, snapshot_root: Path) -> list[str]:
    """Copy the aggregation-level boundary files into the snapshot.

    These cannot be expressed in ``output_data_patterns.txt`` because their names are
    chosen per study (``KE_ADM1_ThreeCounties_AgStats_LongRains.geojson``), so they are
    read out of the run config instead.

    Without them a snapshot is not self-contained: the interactive results map draws
    every non-primary level from these files, and would fall back to whatever the *live*
    study currently holds - so re-preparing a study with different boundaries would
    silently redraw its own history. ESRI shapefiles are copied with their sidecars,
    since a lone ``.shp`` cannot be opened.
    """
    config_path = study_root / "config.yaml"
    if not config_path.is_file():
        return []
    try:
        import yaml as _yaml

        cfg = _yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return []

    levels = ((cfg.get("eval_params") or {}).get("aggregation_levels") or {}).values()
    src_dir = study_root / "aggregation_shapefiles"
    dest = snapshot_root / "aggregation_shapefiles"

    copied: list[str] = []
    for entry in levels:
        name = (entry or {}).get("shapefile")
        if not name:
            continue
        src = Path(str(name))
        if not src.is_absolute():
            src = src_dir / src
        if not src.is_file():
            continue
        # A shapefile is a file set; a GeoJSON is a single file and has no siblings.
        members = sorted(src.parent.glob(src.stem + ".*")) if src.suffix.lower() == ".shp" else [src]
        for member in members:
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / member.name
            if not target.exists():
                shutil.copy2(member, target)
                copied.append(member.name)
    return copied


def zip_snapshot(snapshot_dir: Path, out_zip: Path, patterns: list[str] | None = None) -> int:
    """Zip the snapshot. With ``patterns``, include only files whose basename matches
    one of them - used to build the core bundle from the same snapshot."""
    count = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _dirnames, filenames in os.walk(snapshot_dir):
            for name in filenames:
                if name == RUN_META_FILENAME:
                    continue
                if patterns is not None and not any(fnmatch.fnmatch(name, p) for p in patterns):
                    continue
                full = Path(dirpath) / name
                zf.write(full, full.relative_to(snapshot_dir))
                count += 1
    return count


def rclone_list_dir(target_dir: str) -> list[str]:
    """Return top-level subfolder names under ``target_dir``, or [] if missing."""
    result = subprocess.run(
        ["rclone", "lsf", "--dirs-only", target_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if "directory not found" in stderr or "not found" in stderr:
            return []
        raise RuntimeError(f"rclone lsf failed ({result.returncode}): {result.stderr}")
    return [line.rstrip("/") for line in result.stdout.splitlines() if line.strip()]


def rclone_upload(local_file: Path, remote_dir: str) -> None:
    subprocess.run(["rclone", "copy", "--progress", str(local_file), remote_dir], check=True)


def pick_unique_folder_name(base_name: str, existing: list[str]) -> str:
    if base_name not in existing:
        return base_name
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    used = {int(m.group(1)) for e in existing if (m := pattern.match(e))}
    n = 1
    while n in used:
        n += 1
    return f"{base_name}_{n}"


def join_target(target: str, prefix: str) -> str:
    target = target.rstrip("/")
    prefix = (prefix or "").strip("/")
    return f"{target}/{prefix}" if prefix else target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study-dir", type=Path, required=True, help="sim_study_head_dir of the run")
    ap.add_argument("--study-id", type=str, required=True)
    ap.add_argument("--patterns-file", type=Path, required=True)
    ap.add_argument(
        "--rclone-target",
        type=str,
        default="",
        help="Optional rclone remote (e.g. 'gdrive:' or 'gdrive:vercye_kenya'). "
        "When empty the run is only snapshotted locally.",
    )
    ap.add_argument(
        "--rclone-folder-prefix",
        type=str,
        default="",
        help="Optional subfolder under the rclone target to group uploads",
    )
    ap.add_argument(
        "--marker-file", type=Path, required=True, help="File to touch on success so Snakemake tracks completion"
    )
    args = ap.parse_args()

    root = args.study_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Study dir is not a directory: {root}")

    patterns, core_patterns = load_patterns(args.patterns_file)
    print(
        f"Loaded {len(patterns)} patterns from {args.patterns_file} "
        f"({len(core_patterns)} tagged {CORE_TAG} for the core bundle)"
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_root = root / RUN_RESULTS_DIRNAME / run_id

    # Avoid overwriting an existing snapshot from the same second.
    suffix = 1
    while snapshot_root.exists():
        snapshot_root = root / RUN_RESULTS_DIRNAME / f"{run_id}_{suffix}"
        suffix += 1
    snapshot_root.mkdir(parents=True, exist_ok=False)

    n_copied = copy_matches_to_snapshot(root, patterns, snapshot_root)
    print(f"Snapshotted {n_copied} files into {snapshot_root}")
    if n_copied == 0:
        # Leave nothing behind on an empty snapshot.
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise SystemExit("No files matched the output patterns - refusing to create an empty snapshot")

    n_shapes = snapshot_aggregation_shapefiles(root, snapshot_root)
    print(f"Copied {len(n_shapes)} aggregation shapefile(s) into the snapshot")

    apsim_manifest = snapshot_apsim_sources(root, snapshot_root)
    print(
        f"Persisted {len(apsim_manifest.get('files', []))} APSIM source template(s) "
        f"covering {len(apsim_manifest.get('region_to_template', {}))} region(s)"
    )

    remote_dir = None
    uploaded_zips: list[str] = []
    rclone_target = (args.rclone_target or "").strip()
    if rclone_target:
        if shutil.which("rclone") is None:
            raise SystemExit("rclone is not installed or not on PATH but rclone_target is set")

        target_dir_root = join_target(rclone_target, args.rclone_folder_prefix)
        base_name = f"{args.study_id}_{datetime.now().strftime('%Y%m%d')}"
        existing = rclone_list_dir(target_dir_root)
        folder_name = pick_unique_folder_name(base_name, existing)
        remote_dir = f"{target_dir_root.rstrip('/')}/{folder_name}"
        print(f"Uploading to remote folder: {remote_dir}")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / f"{folder_name}.zip"
            n_zipped = zip_snapshot(snapshot_root, zip_path)
            print(f"Packaged {n_zipped} files into {zip_path.name} ({zip_path.stat().st_size} bytes)")
            rclone_upload(zip_path, remote_dir)
            uploaded_zips.append(zip_path.name)

            # Core bundle: the same snapshot minus the rasters, for collaborators who
            # only need the numbers and the reports. Uploaded into the same folder.
            if core_patterns:
                core_zip_path = Path(tmp) / f"{folder_name}_core.zip"
                n_core = zip_snapshot(snapshot_root, core_zip_path, patterns=core_patterns)
                if n_core:
                    print(
                        f"Packaged {n_core} files into {core_zip_path.name} " f"({core_zip_path.stat().st_size} bytes)"
                    )
                    rclone_upload(core_zip_path, remote_dir)
                    uploaded_zips.append(core_zip_path.name)
                else:
                    print(f"No files matched the {CORE_TAG} patterns - skipping the core bundle")
    else:
        print("No rclone target configured - skipping upload")

    meta = {
        "run_id": snapshot_root.name,
        "study_id": args.study_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "file_count": n_copied,
        "uploaded_to": remote_dir,
        "uploaded_zips": uploaded_zips,
    }
    (snapshot_root / RUN_META_FILENAME).write_text(json.dumps(meta, indent=2))

    args.marker_file.parent.mkdir(parents=True, exist_ok=True)
    args.marker_file.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Run snapshot complete. Marker written to {args.marker_file}")


if __name__ == "__main__":
    main()

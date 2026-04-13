#!/usr/bin/env python3
"""Package VERCYe study outputs and upload them to an rclone remote.

Creates a single zip containing every file under the study directory whose
basename matches a glob in ``output_data_patterns.txt`` and uploads the zip
to ``<rclone_target>/<rclone_folder_prefix>/<study_id>_<YYYYMMDD>[_N]/``
using ``rclone``. The ``_N`` suffix is appended only when a folder with the
same base name is already present on the target, so re-runs on the same day
never overwrite an existing upload.

``rclone_target`` must be (or start with) a configured rclone remote, e.g.
``gdrive:`` or ``gdrive:vercye_kenya``. Configure remotes once with
``rclone config``.
"""

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path


def load_patterns(patterns_file: Path) -> list[str]:
    patterns = []
    for line in patterns_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    if not patterns:
        raise ValueError(f"No patterns loaded from {patterns_file}")
    return patterns


def zip_pattern_matches(root: Path, patterns: list[str], out_zip: Path) -> int:
    count = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if any(fnmatch.fnmatch(name, p) for p in patterns):
                    full = Path(dirpath) / name
                    zf.write(full, full.relative_to(root))
                    count += 1
    return count


def rclone_list_dir(target_dir: str) -> list[str]:
    """Return top-level subfolder names under ``target_dir``, or [] if missing."""
    result = subprocess.run(
        ["rclone", "lsf", "--dirs-only", target_dir],
        capture_output=True, text=True,
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
    ap.add_argument("--rclone-target", type=str, required=True,
                    help="Configured rclone remote, e.g. 'gdrive:' or 'gdrive:vercye_kenya'")
    ap.add_argument("--rclone-folder-prefix", type=str, default="",
                    help="Optional subfolder under the rclone target to group uploads")
    ap.add_argument("--marker-file", type=Path, required=True,
                    help="File to touch on success so Snakemake tracks completion")
    args = ap.parse_args()

    if shutil.which("rclone") is None:
        raise SystemExit("rclone is not installed or not on PATH")

    root = args.study_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Study dir is not a directory: {root}")

    patterns = load_patterns(args.patterns_file)
    print(f"Loaded {len(patterns)} patterns from {args.patterns_file}")

    target_dir_root = join_target(args.rclone_target, args.rclone_folder_prefix)
    base_name = f"{args.study_id}_{datetime.now().strftime('%Y%m%d')}"
    existing = rclone_list_dir(target_dir_root)
    folder_name = pick_unique_folder_name(base_name, existing)
    remote_dir = f"{target_dir_root.rstrip('/')}/{folder_name}"
    print(f"Uploading to remote folder: {remote_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / f"{folder_name}.zip"
        n = zip_pattern_matches(root, patterns, zip_path)
        print(f"Packaged {n} files into {zip_path.name} ({zip_path.stat().st_size} bytes)")
        if n == 0:
            raise SystemExit("No files matched the output patterns — refusing to upload empty archive")
        rclone_upload(zip_path, remote_dir)

    args.marker_file.parent.mkdir(parents=True, exist_ok=True)
    args.marker_file.write_text(f"{remote_dir}\n")
    print(f"Upload complete. Marker written to {args.marker_file}")


if __name__ == "__main__":
    main()

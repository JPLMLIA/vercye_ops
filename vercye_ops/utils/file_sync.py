"""Content-aware file tree sync used by setup_vercye_task.

Kept dependency-free so it can be unit-tested in isolation from the
celery worker module.
"""

import os
import shutil


def files_have_same_content(src: str, dst: str) -> bool:
    """Return True if both files exist and have byte-identical content."""
    try:
        if os.path.getsize(src) != os.path.getsize(dst):
            return False
    except OSError:
        return False

    chunk_size = 64 * 1024
    with open(src, "rb") as fsrc, open(dst, "rb") as fdst:
        while True:
            a = fsrc.read(chunk_size)
            b = fdst.read(chunk_size)
            if a != b:
                return False
            if not a:
                return True


def sync_tree_content_aware(src_root: str, dst_root: str) -> None:
    """Copy src_root into dst_root, but only overwrite destination files
    whose bytes differ from the source. Files that already match are left
    untouched so their mtimes are preserved - this prevents snakemake from
    re-running rules whose inputs did not actually change.

    Files present in dst_root but absent from src_root are left in place
    (matching the previous shutil.copytree(dirs_exist_ok=True) semantics,
    which is required to preserve snakemake intermediate outputs like .db,
    .met, and *_LAI_STATS.csv that live in the same per-region directories).
    """
    for src_dir, _dirs, files in os.walk(src_root):
        rel_dir = os.path.relpath(src_dir, src_root)
        dst_dir = dst_root if rel_dir == "." else os.path.join(dst_root, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in files:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            if os.path.exists(dst_path) and files_have_same_content(src_path, dst_path):
                continue
            shutil.copy2(src_path, dst_path)

"""Tests for vercye_ops.utils.file_sync.sync_tree_content_aware.

The key invariant we care about is that re-syncing a tree where only some
files changed must preserve the mtimes of the unchanged files. This is what
allows snakemake to skip downstream rules whose inputs did not actually
change (e.g. an APSIM-only re-upload should not invalidate LAI/met outputs).
This sync helper is used by both the webapp setup task and the CLI `prep`
mode.
"""

import os
import time

import pytest

from vercye_ops.utils.file_sync import files_have_same_content, sync_tree_content_aware


def _write(path, content: bytes, mtime: float | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _mtime(path):
    return os.stat(path).st_mtime


class TestFilesHaveSameContent:
    def test_identical(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.write_bytes(b"hello world")
        b.write_bytes(b"hello world")
        assert files_have_same_content(str(a), str(b)) is True

    def test_different_size(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.write_bytes(b"hello")
        b.write_bytes(b"hello world")
        assert files_have_same_content(str(a), str(b)) is False

    def test_same_size_different_bytes(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.write_bytes(b"hello")
        b.write_bytes(b"world")
        assert files_have_same_content(str(a), str(b)) is False

    def test_missing_dst(self, tmp_path):
        a = tmp_path / "a"
        a.write_bytes(b"x")
        assert files_have_same_content(str(a), str(tmp_path / "missing")) is False


class TestSyncTreeContentAware:
    @pytest.fixture
    def trees(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        return src, dst

    def test_new_file_is_copied(self, trees):
        src, dst = trees
        _write(str(src / "year/region/region_template.apsimx"), b"apsim-v1")

        sync_tree_content_aware(str(src), str(dst))

        copied = dst / "year/region/region_template.apsimx"
        assert copied.exists()
        assert copied.read_bytes() == b"apsim-v1"

    def test_unchanged_file_keeps_mtime(self, trees):
        """The critical invariant: byte-identical files must NOT be touched,
        otherwise snakemake invalidates downstream rules unnecessarily."""
        src, dst = trees
        old_mtime = time.time() - 10_000  # well in the past

        # Pre-existing file in dst (e.g. a region geojson from a prior setup)
        dst_file = dst / "year/region/region.geojson"
        _write(str(dst_file), b'{"type":"Feature"}', mtime=old_mtime)

        # Source has the same byte content (prepare_yieldstudy regenerated it
        # deterministically from the same shapefile)
        src_file = src / "year/region/region.geojson"
        _write(str(src_file), b'{"type":"Feature"}')

        sync_tree_content_aware(str(src), str(dst))

        assert dst_file.read_bytes() == b'{"type":"Feature"}'
        assert _mtime(str(dst_file)) == pytest.approx(old_mtime, abs=1e-3)

    def test_changed_file_is_overwritten_and_mtime_updated(self, trees):
        src, dst = trees
        old_mtime = time.time() - 10_000

        dst_file = dst / "year/region/region_template.apsimx"
        _write(str(dst_file), b"apsim-v1", mtime=old_mtime)

        src_file = src / "year/region/region_template.apsimx"
        _write(str(src_file), b"apsim-v2-changed")

        sync_tree_content_aware(str(src), str(dst))

        assert dst_file.read_bytes() == b"apsim-v2-changed"
        assert _mtime(str(dst_file)) > old_mtime + 1

    def test_dst_only_files_are_preserved(self, trees):
        """Snakemake intermediates (.db, .met, *_LAI_STATS.csv) live in the
        same per-region directories and must survive a re-sync."""
        src, dst = trees

        # Source only has the template file
        _write(str(src / "year/region/region_template.apsimx"), b"apsim-v1")

        # Destination has additional intermediates from a previous run
        old_mtime = time.time() - 10_000
        intermediate = dst / "year/region/region.db"
        lai_stats = dst / "year/region/region_LAI_STATS.csv"
        weather = dst / "year/region/region_weather.met"
        _write(str(intermediate), b"sqlite-bytes", mtime=old_mtime)
        _write(str(lai_stats), b"date,lai\n2023-01-01,0.5\n", mtime=old_mtime)
        _write(str(weather), b"weather-data", mtime=old_mtime)

        sync_tree_content_aware(str(src), str(dst))

        # All dst-only files survive untouched
        for f in (intermediate, lai_stats, weather):
            assert f.exists()
            assert _mtime(str(f)) == pytest.approx(old_mtime, abs=1e-3)
        # And the new template was copied through
        assert (dst / "year/region/region_template.apsimx").read_bytes() == b"apsim-v1"

    def test_apsim_only_change_does_not_touch_geojson_or_met(self, trees):
        """End-to-end scenario: only the APSIM template changed. The geojson
        and met files (consumed by independent snakemake branches) must keep
        their mtimes so LAI/met rules are not invalidated."""
        src, dst = trees
        old_mtime = time.time() - 10_000

        # Initial state: dst has geojson, met, and v1 of the apsim template
        geojson_dst = dst / "2023/T-0/region1/region1.geojson"
        met_dst = dst / "2023/T-0/region1/region1_weather.met"
        apsim_dst = dst / "2023/T-0/region1/region1_template.apsimx"
        _write(str(geojson_dst), b"GEOJSON", mtime=old_mtime)
        _write(str(met_dst), b"METFILE", mtime=old_mtime)
        _write(str(apsim_dst), b"APSIM-V1", mtime=old_mtime)

        # Re-prepared src: geojson identical (deterministic regen), apsim
        # template changed. Met is dst-only because prepare_yieldstudy does
        # not write it (it is produced by the snakemake fetch_met_data rule).
        _write(str(src / "2023/T-0/region1/region1.geojson"), b"GEOJSON")
        _write(str(src / "2023/T-0/region1/region1_template.apsimx"), b"APSIM-V2")

        sync_tree_content_aware(str(src), str(dst))

        # geojson untouched -> LAI/cropmask/met rules will be skipped
        assert _mtime(str(geojson_dst)) == pytest.approx(old_mtime, abs=1e-3)
        # met untouched -> downstream met-consuming rules will be skipped
        assert _mtime(str(met_dst)) == pytest.approx(old_mtime, abs=1e-3)
        assert met_dst.read_bytes() == b"METFILE"
        # apsim template was actually updated
        assert apsim_dst.read_bytes() == b"APSIM-V2"
        assert _mtime(str(apsim_dst)) > old_mtime + 1

    def test_nested_directories_created_as_needed(self, trees):
        src, dst = trees
        _write(str(src / "a/b/c/d/file.txt"), b"deep")

        sync_tree_content_aware(str(src), str(dst))

        assert (dst / "a/b/c/d/file.txt").read_bytes() == b"deep"

    def test_no_changes_at_all_is_a_noop(self, trees):
        """Re-submitting setup with no changes should leave every mtime
        intact (so `snakemake -n` reports nothing to do)."""
        src, dst = trees
        old_mtime = time.time() - 10_000

        files = {
            "year/r1/r1.geojson": b"G1",
            "year/r1/r1_template.apsimx": b"A1",
            "year/r2/r2.geojson": b"G2",
            "year/r2/r2_template.apsimx": b"A2",
            "run_config.yaml": b"key: value\n",
        }
        for rel, content in files.items():
            _write(str(src / rel), content)
            _write(str(dst / rel), content, mtime=old_mtime)

        sync_tree_content_aware(str(src), str(dst))

        for rel in files:
            assert _mtime(str(dst / rel)) == pytest.approx(old_mtime, abs=1e-3)

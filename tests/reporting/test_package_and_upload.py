"""Tests for the snapshot/upload packaging, in particular the core data bundle.

The full bundle is dominated by rasters - on the three-county Kenya study it was
2.8 GiB zipped, of which `aggregated_LAI_MAX_*.tif` and `yield_mosaic_projected_*.tif`
alone accounted for 3.4 GiB raw. Collaborators who only need the numbers and the
reports should not have to pull that, so patterns tagged `@core` in
`output_data_patterns.txt` are additionally packaged into a second, small zip.

The tag is a subset marker on the existing lines rather than a separate patterns
file: the core bundle is by definition a subset of what is snapshotted, and this
keeps one source of truth and needs no run-config change for existing studies.
"""

import importlib.util
import zipfile
from pathlib import Path

import pytest

REPORTING = Path(__file__).resolve().parents[2] / "vercye_ops" / "reporting"
PATTERNS_FILE = REPORTING / "output_data_patterns.txt"


def _load(name):
    path = REPORTING / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pkg():
    return _load("package_and_upload")


def _write(tmp_path, name, text="x"):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


class TestLoadPatterns:
    def test_untagged_file_yields_no_core_patterns(self, pkg, tmp_path):
        """A patterns file with no @core tags must behave exactly as before."""
        f = _write(tmp_path, "p.txt", "a_*.csv\n# comment\n\nb_*.pdf\n")
        all_p, core_p = pkg.load_patterns(f)
        assert all_p == ["a_*.csv", "b_*.pdf"]
        assert core_p == []

    def test_tagged_patterns_appear_in_both_lists(self, pkg, tmp_path):
        """A @core pattern is still part of the full bundle - core is a subset."""
        f = _write(tmp_path, "p.txt", "a_*.csv @core\nb_*.tif\nc_*.pdf   @core\n")
        all_p, core_p = pkg.load_patterns(f)
        assert all_p == ["a_*.csv", "b_*.tif", "c_*.pdf"]
        assert core_p == ["a_*.csv", "c_*.pdf"]

    def test_tag_is_stripped_from_the_glob(self, pkg, tmp_path):
        """The tag must not leak into the glob or it would match nothing."""
        f = _write(tmp_path, "p.txt", "a_*.csv @core\n")
        all_p, core_p = pkg.load_patterns(f)
        assert "@core" not in all_p[0] and "@core" not in core_p[0]

    def test_empty_file_still_raises(self, pkg, tmp_path):
        f = _write(tmp_path, "p.txt", "# only a comment\n")
        with pytest.raises(ValueError):
            pkg.load_patterns(f)


class TestZipSnapshot:
    def _snapshot(self, tmp_path):
        root = tmp_path / "snap"
        _write(root, "all_predictions_s_ADM1_T-0.csv")
        _write(root, "2019/T-0/final_report_s_2019_T-0.pdf")
        _write(root, "2019/T-0/aggregated_LAI_MAX_s_2019_T-0.tif", "x" * 5000)
        _write(root, "2019/T-0/yield_mosaic_projected_s_2019_T-0.tif", "x" * 5000)
        _write(root, "apsim/Template.apsimx")
        _write(root, pkg_meta := "_run_meta.json")
        return root, pkg_meta

    def test_unfiltered_zip_contains_everything_but_meta(self, pkg, tmp_path):
        root, _ = self._snapshot(tmp_path)
        out = tmp_path / "full.zip"
        n = pkg.zip_snapshot(root, out)
        names = zipfile.ZipFile(out).namelist()
        assert n == len(names) == 5
        assert "_run_meta.json" not in names
        assert "apsim/Template.apsimx" in names

    def test_core_zip_contains_only_tagged_patterns(self, pkg, tmp_path):
        root, _ = self._snapshot(tmp_path)
        out = tmp_path / "core.zip"
        n = pkg.zip_snapshot(root, out, patterns=["all_predictions_*.csv", "final_report_*.pdf"])
        names = sorted(zipfile.ZipFile(out).namelist())
        assert n == 2
        assert names == ["2019/T-0/final_report_s_2019_T-0.pdf", "all_predictions_s_ADM1_T-0.csv"]

    def test_core_zip_excludes_the_heavy_rasters(self, pkg, tmp_path):
        """The whole point of the core bundle: no rasters."""
        root, _ = self._snapshot(tmp_path)
        out = tmp_path / "core.zip"
        pkg.zip_snapshot(root, out, patterns=["all_predictions_*.csv", "final_report_*.pdf"])
        assert not [n for n in zipfile.ZipFile(out).namelist() if n.endswith(".tif")]

    def test_core_zip_is_smaller_than_full(self, pkg, tmp_path):
        root, _ = self._snapshot(tmp_path)
        full, core = tmp_path / "f.zip", tmp_path / "c.zip"
        pkg.zip_snapshot(root, full)
        pkg.zip_snapshot(root, core, patterns=["all_predictions_*.csv"])
        assert core.stat().st_size < full.stat().st_size


class TestShippedPatternsFile:
    """Guard the actual file the pipeline ships with."""

    def test_core_subset_is_tagged_and_is_a_strict_subset(self, pkg):
        all_p, core_p = pkg.load_patterns(PATTERNS_FILE)
        assert core_p, "expected @core tags in the shipped patterns file"
        assert set(core_p) <= set(all_p)
        assert len(core_p) < len(all_p)

    def test_core_holds_the_csvs_reports_and_multiyear_report(self, pkg):
        _, core_p = pkg.load_patterns(PATTERNS_FILE)
        for expected in [
            "all_predictions_*.csv",
            "referencedata_*.csv",
            "agg_yield_estimates_*.csv",
            "final_report_*.pdf",
            "multiyear_summary*.zip",
            "lai_report_*.pdf",
        ]:
            assert expected in core_p, f"{expected} should be in the core bundle"

    def test_core_excludes_rasters_and_interactive_maps(self, pkg):
        _, core_p = pkg.load_patterns(PATTERNS_FILE)
        for excluded in [
            "aggregated_LAI_MAX_*.tif",
            "yield_mosaic_projected_*.tif",
            "aggregated_cropmask_*.tif",
            "interactive_map_*.zip",
        ]:
            assert excluded not in core_p, f"{excluded} must stay out of the core bundle"

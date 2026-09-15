"""Regression tests: aggregation-level names must be matched exactly, not by prefix.

The on-disk name is `agg_yield_estimates_{level}_{study_id}_{year}_{timepoint}.csv`.
Globbing `agg_yield_estimates_{level}_*_{year}_{timepoint}.csv` lets the wildcard
swallow a *different* level whose name merely starts with this one, because both
level names and study ids may contain underscores. With levels `ADM1` and
`ADM1_ThreeCounties` that mis-match aborted the multiyear aggregation, and in the
final report (which took `matching_files[0]`) it would have silently built one
level's section from another level's numbers.
"""

import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest

REPORTING = Path(__file__).resolve().parents[2] / "vercye_ops" / "reporting"
STUDY = "kenya-longrains"
YEAR = "2026"
TP = "T-0"
COLLIDING_LEVELS = ["ADM0", "ADM1", "ADM1_ThreeCounties", "ADM2"]


def _load(name):
    path = REPORTING / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def study_tree(tmp_path):
    """A study dir carrying one agg_yield_estimates file per level, each tagged
    with its own level name so a mis-match is detectable in the contents."""
    tp_dir = tmp_path / YEAR / TP
    tp_dir.mkdir(parents=True)
    for lvl in COLLIDING_LEVELS:
        pd.DataFrame({"region": [f"r_{lvl}"], "mean_yield_kg_ha": [1000.0], "level": [lvl]}).to_csv(
            tp_dir / f"agg_yield_estimates_{lvl}_{STUDY}_{YEAR}_{TP}.csv", index=False
        )
    return tmp_path


class TestCollectFilesExactMatch:
    def test_adm1_does_not_pick_up_adm1_threecounties(self, study_tree):
        m = _load("aggregate_multiyear_predictions")
        preds, _ = m.collect_files(str(study_tree), "ADM1", TP, STUDY)
        assert list(preds) == [YEAR]
        assert pd.read_csv(preds[YEAR])["level"].iloc[0] == "ADM1"

    def test_each_level_resolves_to_its_own_file(self, study_tree):
        m = _load("aggregate_multiyear_predictions")
        for lvl in COLLIDING_LEVELS:
            preds, _ = m.collect_files(str(study_tree), lvl, TP, STUDY)
            assert pd.read_csv(preds[YEAR])["level"].iloc[0] == lvl, f"{lvl} resolved to the wrong file"

    def test_absent_level_returns_nothing(self, study_tree):
        m = _load("aggregate_multiyear_predictions")
        preds, _ = m.collect_files(str(study_tree), "ADM9", TP, STUDY)
        assert preds == {}


class TestFinalReportExactMatch:
    def test_prefix_pins_the_level(self, study_tree):
        """Mirrors the final report's selection: the study_id must follow the level
        name immediately, so ADM1 cannot select the ADM1_ThreeCounties file."""
        regions_dir = study_tree / YEAR / TP
        import glob as _glob

        for lvl in COLLIDING_LEVELS:
            prefix = f"agg_yield_estimates_{lvl}_{STUDY}_"
            matches = [
                f
                for f in _glob.glob(os.path.join(str(regions_dir), f"agg_yield_estimates_{lvl}_*.csv"))
                if os.path.basename(f).startswith(prefix)
            ]
            assert len(matches) == 1, f"{lvl} matched {len(matches)} files"
            assert pd.read_csv(matches[0])["level"].iloc[0] == lvl

    def test_naive_glob_would_have_collided(self, study_tree):
        """Guards the guard: without the prefix pin, ADM1 really does match two files."""
        import glob as _glob

        naive = _glob.glob(os.path.join(str(study_tree / YEAR / TP), "agg_yield_estimates_ADM1_*.csv"))
        assert len(naive) == 2

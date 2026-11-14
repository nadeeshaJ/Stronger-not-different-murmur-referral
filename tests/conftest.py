import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from murmur.paths import dataset_dir, derived_dir, load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config("phase1")


@pytest.fixture(scope="session")
def cohort(cfg):
    ds = dataset_dir(cfg)
    if not (ds / "training_data.csv").exists():
        pytest.skip(f"CirCor not found at {ds}")
    from murmur.cohort import build_cohort
    return build_cohort(ds, cfg)


@pytest.fixture(scope="session")
def folds(cohort, cfg):
    from murmur.partitions import make_partitions
    pc = cfg["partitions"]
    return make_partitions(cohort[0], pc["seeds"], pc["n_outer"], pc["n_inner"])

"""Shared fixtures: small synthetic panels so tests never need the real database or data files."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def wide_panel() -> pd.DataFrame:
    """120 days x 2 jurisdictions x 2 groups of Poisson counts with a weekly pattern."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-05", periods=120, freq="D")
    cols = pd.MultiIndex.from_tuples([(1, "PATRIMONIO"), (1, "PERSONA"), (2, "PATRIMONIO"), (2, "PERSONA")])
    base = np.array([6.0, 3.0, 4.0, 2.0])
    weekly = 1 + 0.3 * np.sin(2 * np.pi * idx.dayofweek.to_numpy() / 7)[:, None]
    return pd.DataFrame(rng.poisson(base * weekly), index=idx, columns=cols).astype(float)

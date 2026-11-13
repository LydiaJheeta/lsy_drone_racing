"""pytest setup: import lsy_drone_racing before scipy, make project/ modules importable."""

from __future__ import annotations

import lsy_drone_racing  # noqa: F401  (must precede scipy: crazyflow sets SCIPY_ARRAY_API)

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def real_env():
    """The real level-0 attitude env (created once; JIT compilation takes a while)."""
    from common import load_cfg, make_env

    cfg = load_cfg("att_level0.toml")
    env = make_env(cfg)
    yield cfg, env
    env.close()

"""Tests for diagnose.py."""

from __future__ import annotations

import sys

import pytest

from diagnose import diagnose


def test_refuses_to_reuse_an_existing_output_folder(tmp_path):
    before = sys.stdout
    with pytest.raises(FileExistsError):
        diagnose(levels=[0], n_runs=1, out_dir=str(tmp_path))  # tmp_path already exists
    assert sys.stdout is before  # failed before any stdout redirection
    assert not list(tmp_path.iterdir())  # nothing written

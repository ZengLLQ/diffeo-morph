"""Unit tests for DiffeoMorph CLI."""

import subprocess
import sys


def test_cli_help():
    res = subprocess.run([sys.executable, "-m", "diffeo_morph.cli", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "DiffeoMorph" in res.stdout


def test_cli_register():
    res = subprocess.run(
        [sys.executable, "-m", "diffeo_morph.cli", "register", "--size", "16", "--iterations", "10"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Stationary Velocity Field (SVF) Registration" in res.stdout
    assert "Diffeomorphic" in res.stdout


def test_cli_alzheimer():
    res = subprocess.run(
        [sys.executable, "-m", "diffeo_morph.cli", "alzheimer", "--atrophy-severity", "1.2"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Longitudinal Brain MRI Morphometry & TBM Report" in res.stdout
    assert "DIAGNOSTIC STAGING EVALUATION" in res.stdout

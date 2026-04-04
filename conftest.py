"""
tests/conftest.py
=================
Shared pytest configuration, fixtures and helpers for the noawclg test suite.
"""

from __future__ import annotations

import logging

import pytest


# ── silence library noise during tests ───────────────────────────────────────
logging.getLogger("cfgrib").setLevel(logging.CRITICAL)
logging.getLogger("gfs_dataset").setLevel(logging.CRITICAL)


# ── global markers ────────────────────────────────────────────────────────────
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so they appear in --markers output."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require a real network connection "
        "(deselect with -m 'not integration')",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests that are computationally expensive",
    )
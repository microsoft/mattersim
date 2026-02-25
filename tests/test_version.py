"""
Tests for __version__.py (GitHub issue #140):
Mattersim failed to import with setuptools>=82 due to removed pkg_resources.
"""

import importlib
import sys
from unittest import mock


def _reimport_version_module():
    """Force a fresh import of mattersim.__version__."""
    for mod_name in list(sys.modules):
        if mod_name.startswith("mattersim"):
            del sys.modules[mod_name]
    return importlib.import_module("mattersim.__version__")


def test_version_available():
    """mattersim.__version__ is a non-empty string."""
    import mattersim

    assert isinstance(mattersim.__version__, str)
    assert mattersim.__version__


def test_import_succeeds_without_pkg_resources():
    """Import must work when pkg_resources is absent (setuptools>=82, issue #140)."""
    for mod_name in list(sys.modules):
        if mod_name.startswith("mattersim"):
            del sys.modules[mod_name]

    with mock.patch.dict(sys.modules, {"pkg_resources": None}):
        mod = importlib.import_module("mattersim.__version__")
        assert isinstance(mod.__version__, str)
        assert mod.__version__

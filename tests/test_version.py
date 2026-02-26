#!/usr/bin/env python
import importlib

from vit import __version__


def test_version():
    assert isinstance(__version__, str)
    assert __version__

    try:
        version_module = importlib.import_module("vit._version")
    except ModuleNotFoundError:
        return

    assert isinstance(version_module.__version__, str)
    assert __version__ == version_module.__version__

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from vit import __version__
from vit._version import __version__ as __version__2


def test_version():
    assert isinstance(__version__, str)
    assert isinstance(__version__2, str)
    assert __version__ == __version__2

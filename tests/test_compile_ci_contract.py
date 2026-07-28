"""CI selection contract for torch.compile coverage."""

from __future__ import annotations

import pytest

from tests import test_checkpointing


def test_checkpointing_compile_regression_uses_compile_marker() -> None:
    compile_test = test_checkpointing.TestActivationCheckpointing.test_checkpointing_with_torch_compile
    marks = getattr(compile_test, "pytestmark", ())

    assert any(isinstance(mark, pytest.Mark) and mark.name == "compile" for mark in marks)
    assert not any(isinstance(mark, pytest.Mark) and mark.name == "ci_skip" for mark in marks)

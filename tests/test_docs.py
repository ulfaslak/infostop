"""Run the `>>>` examples in the docs pages and compare their output."""

import doctest
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


@pytest.mark.parametrize("page", ["about.rst", "usage/usage.rst"])
def test_docs_examples(page):
    result = doctest.testfile(str(DOCS_DIR / page), module_relative=False)
    assert result.attempted > 0, f"no examples found in {page}"
    assert result.failed == 0, f"{result.failed} example(s) in {page} failed"

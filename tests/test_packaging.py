import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "setup.py")), reason="needs the source tree"
)
def test_packages_only_infostop_and_cpputils(tmp_path):
    pytest.importorskip("setuptools")
    # Write fresh metadata from `setup.py` into `tmp_path` and read the top-level
    # names a wheel would install. Installed or leftover metadata cannot interfere.
    subprocess.run(
        [sys.executable, "setup.py", "-q", "egg_info", "--egg-base", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    top_level = (tmp_path / "infostop.egg-info" / "top_level.txt").read_text().split()
    assert sorted(top_level) == ["cpputils", "infostop"]

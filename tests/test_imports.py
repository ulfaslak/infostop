import os
import subprocess
import sys

import infostop

# Each check imports infostop in a fresh interpreter: warnings raised while a module
# is first imported do not repeat once it is already imported, and this test session
# imports infostop. The directory holding the infostop under test is put first on
# `sys.path`, so the subprocess cannot pick up another installed copy (e.g. with
# PYTHONSAFEPATH=1).
PRELUDE = """
import sys
sys.path.insert(0, sys.argv[1])
"""

IMPORT_WITH_DEPRECATIONS_AS_ERRORS = PRELUDE + """
import warnings
for category in (DeprecationWarning, PendingDeprecationWarning):
    warnings.filterwarnings("error", category=category, module="infostop")
import infostop
print(infostop.__file__)
"""

# SciPy < 1.8 has `QhullError` only in `scipy.spatial.qhull`
IMPORT_WITH_OLD_SCIPY_LAYOUT = PRELUDE + """
import types
import scipy.spatial
class OldQhullError(Exception):
    pass
del scipy.spatial.QhullError
sys.modules["scipy.spatial.qhull"] = types.ModuleType("scipy.spatial.qhull")
sys.modules["scipy.spatial.qhull"].QhullError = OldQhullError
import infostop
assert infostop.utils.QhullError is OldQhullError
print(infostop.__file__)
"""


def _import_in_subprocess(script):
    root = os.path.dirname(os.path.dirname(os.path.abspath(infostop.__file__)))
    result = subprocess.run(
        [sys.executable, "-c", script, root], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert os.path.samefile(result.stdout.strip(), infostop.__file__)


def test_import_warns_no_deprecation():
    _import_in_subprocess(IMPORT_WITH_DEPRECATIONS_AS_ERRORS)


def test_import_falls_back_to_old_scipy_layout():
    _import_in_subprocess(IMPORT_WITH_OLD_SCIPY_LAYOUT)

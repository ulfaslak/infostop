import os
import subprocess
import sys

import infostop

# Import in a fresh interpreter: warnings raised while a module is first imported do
# not repeat once it is already imported, and this test session imports infostop.
# The directory holding the infostop under test is put first on `sys.path`, so the
# subprocess cannot pick up another installed copy (e.g. with PYTHONSAFEPATH=1).
IMPORT_WITH_DEPRECATIONS_AS_ERRORS = """
import sys
import warnings
sys.path.insert(0, sys.argv[1])
for category in (DeprecationWarning, PendingDeprecationWarning):
    warnings.filterwarnings("error", category=category, module="infostop")
import infostop
print(infostop.__file__)
"""


def test_import_warns_no_deprecation():
    root = os.path.dirname(os.path.dirname(os.path.abspath(infostop.__file__)))
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_WITH_DEPRECATIONS_AS_ERRORS, root],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert os.path.samefile(result.stdout.strip(), infostop.__file__)

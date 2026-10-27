import os
import subprocess
import sys

import infostop

# Import in a fresh interpreter: warnings raised while a module is first imported do
# not repeat once it is already imported, and this test session imports infostop.
IMPORT_WITH_DEPRECATIONS_AS_ERRORS = """
import warnings
warnings.filterwarnings("error", category=DeprecationWarning, module="infostop")
import infostop
"""


def test_import_uses_no_deprecated_api():
    # Run next to the infostop under test, so the subprocess imports the same copy
    root = os.path.dirname(os.path.dirname(os.path.abspath(infostop.__file__)))
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_WITH_DEPRECATIONS_AS_ERRORS],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

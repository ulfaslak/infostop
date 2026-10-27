from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import zipfile


def test_wheel_excludes_tests_but_sdist_keeps_them(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    source_root = tmp_path / "source"
    source_root.mkdir()

    for filename in ("LICENSE", "MANIFEST.in", "README.md", "setup.py"):
        shutil.copy2(repo_root / filename, source_root / filename)
    for directory in ("cpputils", "infostop", "tests"):
        shutil.copytree(
            repo_root / directory,
            source_root / directory,
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    dist_dir = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--outdir",
            str(dist_dir),
        ],
        cwd=source_root,
        check=True,
    )

    wheel_path = next(dist_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_files = set(wheel.namelist())

    assert "infostop/__init__.py" in wheel_files
    assert any(name.startswith("cpputils.") for name in wheel_files)
    assert not any(name.startswith("tests/") for name in wheel_files)

    sdist_path = next(dist_dir.glob("*.tar.gz"))
    with tarfile.open(sdist_path, "r:gz") as sdist:
        sdist_files = [PurePosixPath(name).parts for name in sdist.getnames()]

    assert any(parts[-2:] == ("tests", "test_models.py") for parts in sdist_files)

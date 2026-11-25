"""Downloads only the .rst sources of the Godot docs repository.

The full repo is large because of its images, so we use a partial clone
(--filter=blob:none) plus a sparse checkout: git fetches the content of
.rst files only.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

from godot_rag.config import Settings


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def _force_remove(func, path, _exc) -> None:
    # git marks its object files read-only, which makes rmtree fail on Windows.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def download_docs(settings: Settings, force: bool = False) -> Path:
    dest = settings.raw_dir
    if dest.exists():
        if not force:
            print(f"Docs already downloaded in {dest} (use --force-download to refresh).")
            return dest
        shutil.rmtree(dest, onexc=_force_remove)

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Godot docs ({settings.docs_branch}) into {dest} ...")
    _git(
        "clone", "--depth", "1", "--branch", settings.docs_branch,
        "--filter=blob:none", "--no-checkout", settings.docs_repo, str(dest),
    )
    # Non-cone sparse patterns use .gitignore syntax: "/dir/**/*.rst" = every .rst under dir.
    patterns = [f"/{folder}/**/*.rst" for folder in settings.docs_include]
    _git("sparse-checkout", "set", "--no-cone", *patterns, cwd=dest)
    _git("checkout", settings.docs_branch, cwd=dest)
    return dest


def iter_rst_files(settings: Settings) -> list[Path]:
    root = settings.raw_dir
    files: list[Path] = []
    for folder in settings.docs_include:
        files.extend(sorted((root / folder).rglob("*.rst")))
    return files

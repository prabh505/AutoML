"""Bundle helpers for tests: unzip, and run `pipeline.py` with `tabpilot` hidden (P14/P15)."""

from __future__ import annotations

import site
import subprocess
import sys
import zipfile
from pathlib import Path


def unzip(path: Path, where: Path) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(where)
    return where


def isolated(bundle: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run `pipeline.py` with `-I -S`: only the stdlib, the bundle dir and third-party
    site-packages are importable. The editable-install `.pth` that exposes `src/` (and so
    `tabpilot`) is never processed.
    """
    packages = [p for p in site.getsitepackages() if Path(p).is_dir()]
    launcher = (
        "import sys, runpy\n"
        "sys.path[:] = [p for p in sys.path if 'site-packages' not in p] + "
        f"{packages!r}\n"
        "try:\n    import tabpilot\n    print('TABPILOT IMPORTABLE', tabpilot.__file__)\n"
        "except ImportError:\n    print('tabpilot not importable')\n"
        "print('sys.path =', sys.path)\n"
        f"sys.argv = ['pipeline.py', *{list(args)!r}]\n"
        "try:\n    runpy.run_path('pipeline.py', run_name='__main__')\n"
        "finally:\n    rt = sys.modules.get('tabpilot_runtime')\n"
        "    print('runtime from', rt.__file__ if rt else None, flush=True)\n"
    )  # fmt: skip
    return subprocess.run([sys.executable, "-I", "-S", "-c", launcher], cwd=bundle,
                          capture_output=True, text=True, timeout=300, check=False)  # fmt: skip

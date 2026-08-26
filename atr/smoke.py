"""Install-verification smoke test for ATR.

Runs in seconds, requires no external credentials or large model weights.
Tier-1 checks only: for end-to-end correctness see the per-recipe commands
in README.md.

Exit code 0 if all checks pass, 1 otherwise.

Usage:
    python -m atr.smoke              # run all checks
    python -m atr.smoke --verbose    # show details for each check
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Tuple


_REPO = Path(__file__).resolve().parent.parent


def _check_imports() -> Tuple[bool, str]:
    modules = [
        "atr.config", "atr.prompt", "atr.build_index", "atr.evaluate",
        "atr.online.main", "atr.online.decomposer", "atr.online.router",
        "atr.online.value_linker", "atr.online.constrained_sql", "atr.online.verifier",
        "atr.offline.multiview_index",
        "atr.clients.chat_utils", "atr.clients.sql_tool",
        "atr.clients.tool_utils",
        "atr.baselines.naive_llm", "atr.baselines.naiverag",
        "atr.baselines.naiverag_sc", "atr.baselines.react",
        "atr.tools.train_router", "atr.tools.download_router",
    ]
    missing = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as exc:
            missing.append(f"{m}: {exc.__class__.__name__}: {exc}")
    if missing:
        return False, "Failed imports:\n  " + "\n  ".join(missing)
    return True, f"all {len(modules)} modules import"


def _check_cli_help() -> Tuple[bool, str]:
    entries = ["atr.build_index", "atr.online.main", "atr.tools.train_router",
               "atr.tools.download_router", "atr.evaluate"]
    failed = []
    for ep in entries:
        try:
            r = subprocess.run(
                [sys.executable, "-m", ep, "--help"],
                capture_output=True, text=True, timeout=20,
                cwd=str(_REPO),
            )
            if r.returncode != 0:
                failed.append(f"{ep}: exit {r.returncode}: {r.stderr.strip()[:150]}")
        except Exception as exc:
            failed.append(f"{ep}: {exc}")
    if failed:
        return False, "Failed --help:\n  " + "\n  ".join(failed)
    return True, f"all {len(entries)} CLIs respond to --help"


def _check_config_backbones() -> Tuple[bool, str]:
    from atr import config
    if not isinstance(config.config_mapping, dict) or not config.config_mapping:
        return False, "config_mapping empty"
    n = len(config.config_mapping)
    keys = sorted(config.config_mapping.keys())[:5]
    return True, f"{n} backbones registered (first: {', '.join(keys)}...)"


def _check_credentials_guard() -> Tuple[bool, str]:
    """Verify .gitignore and pre-commit hook block credentials."""
    gi = (_REPO / ".gitignore").read_text() if (_REPO / ".gitignore").exists() else ""
    needed = ["vertexai.json", "database_config.json"]
    missing = [n for n in needed if n not in gi]
    if missing:
        return False, ".gitignore missing patterns: " + ", ".join(missing)
    hook = _REPO / ".githooks" / "pre-commit"
    if not hook.exists():
        return False, ".githooks/pre-commit not installed"
    if not os.access(hook, os.X_OK):
        return False, ".githooks/pre-commit not executable"
    return True, ".gitignore + pre-commit hook present"


def _check_release_artifacts() -> Tuple[bool, str]:
    must_have = ["LICENSE", "README.md", "requirements.txt"]
    missing = [p for p in must_have if not (_REPO / p).exists()]
    if missing:
        return False, "Missing: " + ", ".join(missing)
    return True, "LICENSE + README + requirements all present"


CHECKS: List[Tuple[str, Callable[[], Tuple[bool, str]]]] = [
    ("module imports",       _check_imports),
    ("CLI --help",           _check_cli_help),
    ("config.config_mapping", _check_config_backbones),
    ("credentials guard",    _check_credentials_guard),
    ("release artifacts",    _check_release_artifacts),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="ATR install-verification smoke test")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-check details even on pass.")
    args = parser.parse_args()

    print(f"Running {len(CHECKS)} smoke checks against {_REPO} ...")
    print()
    failed = []
    for name, fn in CHECKS:
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, f"{exc.__class__.__name__}: {exc}"
        glyph = "✅" if ok else "❌"
        suffix = f"  ({msg})" if (args.verbose or not ok) else ""
        print(f"  {glyph}  {name}{suffix}")
        if not ok:
            failed.append(name)

    print()
    if failed:
        print(f"❌  {len(failed)} of {len(CHECKS)} checks FAILED: {', '.join(failed)}")
        print("    Re-run with --verbose for full details, or check the docs for the failing area.")
        sys.exit(1)
    print(f"✅  all {len(CHECKS)} checks passed. You're good to go.")
    print("    Next: see 'Reproducing the paper' in README.md for end-to-end recipes.")


if __name__ == "__main__":
    main()

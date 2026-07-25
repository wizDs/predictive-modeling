#!/usr/bin/env python3
"""Compute which apps/* are affected by a diff, for the `changes` job in ci.yml.

Changed apps/** files are mapped to the app they're under by walking up to the nearest
pyproject.toml, instead of hand-listing every app in a filter + a matrix + an if-chain
(three places to update -- and forget -- every time an app is added, renamed, or nested,
which this repo has already done more than once). No app name is hardcoded anywhere;
ci.yml's mypy-apps matrix just becomes whatever this script discovers.

Changed src/**-path files need the *other* direction: apps consume src/* packages via uv
workspace deps, not by path, so a change confined to src/budget (say) would never touch an
apps/** path and could never be detected that way. To catch that, every package's
[tool.uv.sources] path=/workspace=true entries are read to build a reverse-dependency graph
from a src/<pkg> to the src packages and apps that depend on it -- transitively, since e.g.
src/shared depends on src/evaluation via workspace=true. See
https://github.com/wizDs/predictive-modeling/issues/16.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import deque
from pathlib import Path

CI_WORKFLOW_PATH = ".github/workflows/ci.yml"


def parse_src_deps(pyproject: Path, repo_root: Path) -> set[str]:
    """The src/<pkg> names `pyproject` depends on via its [tool.uv.sources] table.

    A `path = "..."` entry is resolved relative to `pyproject` and kept only if it lands
    under src/. A `workspace = true` entry is kept by its dependency key as-is -- workspace
    members are always named after their src/<pkg> directory in this repo, and workspace=true
    carries no path of its own to resolve.
    """
    data = tomllib.loads(pyproject.read_text())
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    deps: set[str] = set()
    for key, spec in sources.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("workspace"):
            deps.add(key)
            continue
        path = spec.get("path")
        if not path:
            continue
        resolved = (pyproject.parent / path).resolve()
        try:
            rel_parts = resolved.relative_to(repo_root.resolve()).parts
        except ValueError:
            continue
        if len(rel_parts) >= 2 and rel_parts[0] == "src":
            deps.add(rel_parts[1])
    return deps


def build_reverse_deps(repo_root: Path) -> dict[str, set[str]]:
    """reverse_deps[pkg] -> ids ("src:<pkg>" or "app:<dir>") that depend on src/<pkg>."""
    reverse_deps: dict[str, set[str]] = {}

    def add_edges(pyproject: Path, dependent_id: str) -> None:
        for dep in parse_src_deps(pyproject, repo_root):
            reverse_deps.setdefault(dep, set()).add(dependent_id)

    for pyproject in sorted(repo_root.glob("src/*/pyproject.toml")):
        add_edges(pyproject, f"src:{pyproject.parent.name}")

    apps_root = repo_root / "apps"
    app_pyprojects = sorted(apps_root.glob("*/pyproject.toml")) + sorted(
        apps_root.glob("*/*/pyproject.toml")
    )
    for pyproject in app_pyprojects:
        app_dir = pyproject.parent.relative_to(apps_root).as_posix()
        add_edges(pyproject, f"app:{app_dir}")

    return reverse_deps


def apps_transitively_affected_by(
    changed_src_pkgs: set[str], reverse_deps: dict[str, set[str]]
) -> set[str]:
    """BFS every changed src/<pkg> through reverse_deps to every app that depends on it."""
    apps: set[str] = set()
    visited: set[str] = set()
    queue: deque[str] = deque(f"src:{pkg}" for pkg in changed_src_pkgs)
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        kind, _, name = current.partition(":")
        if kind != "src":
            continue
        for dependent_id in reverse_deps.get(name, ()):
            queue.append(dependent_id)
            if dependent_id.startswith("app:"):
                apps.add(dependent_id.removeprefix("app:"))
    return apps


def app_dirs_for_changed_files(changed_files: list[str], repo_root: Path) -> set[str]:
    """Map each changed apps/** file to its app dir by walking up to the nearest pyproject.toml."""
    apps_root = repo_root / "apps"
    apps: set[str] = set()
    for f in changed_files:
        directory = (repo_root / f).parent
        while directory != apps_root and apps_root in directory.parents:
            if (directory / "pyproject.toml").is_file():
                apps.add(directory.relative_to(apps_root).as_posix())
                break
            directory = directory.parent
    return apps


def all_app_dirs(repo_root: Path) -> set[str]:
    apps_root = repo_root / "apps"
    pyprojects = list(apps_root.glob("*/pyproject.toml")) + list(
        apps_root.glob("*/*/pyproject.toml")
    )
    return {p.parent.relative_to(apps_root).as_posix() for p in pyprojects}


def compute_affected_apps(
    repo_root: Path, apps_files: list[str], src_files: list[str]
) -> set[str]:
    if CI_WORKFLOW_PATH in apps_files:
        # A CI logic change could affect any app regardless of which paths it touches --
        # run every app that currently exists, not just the ones in this diff.
        return all_app_dirs(repo_root)

    affected = app_dirs_for_changed_files(apps_files, repo_root)

    changed_src_pkgs = {
        Path(f).parts[1] for f in src_files if Path(f).parts[:1] == ("src",) and len(Path(f).parts) >= 2
    }
    reverse_deps = build_reverse_deps(repo_root)
    affected |= apps_transitively_affected_by(changed_src_pkgs, reverse_deps)

    return affected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--apps-files", required=True, help="JSON array of changed apps/** paths")
    parser.add_argument("--src-files", required=True, help="JSON array of changed src/** paths")
    args = parser.parse_args()

    affected = compute_affected_apps(
        args.repo_root.resolve(),
        json.loads(args.apps_files),
        json.loads(args.src_files),
    )
    print(f"apps={json.dumps(sorted(affected))}")


if __name__ == "__main__":
    main()

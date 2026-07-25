from pathlib import Path

from affected_apps import CI_WORKFLOW_PATH, compute_affected_apps

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_pyproject(path: Path, sources: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = '[project]\nname = "x"\n'
    if sources:
        body += f"\n[tool.uv.sources]\n{sources}\n"
    path.write_text(body)


def test_direct_apps_file_change_is_unaffected_by_src_logic(tmp_path: Path) -> None:
    write_pyproject(tmp_path / "apps/finance/pyproject.toml")
    write_pyproject(tmp_path / "apps/churn/pyproject.toml")

    affected = compute_affected_apps(
        tmp_path, apps_files=["apps/finance/app.py"], src_files=[]
    )

    assert affected == {"finance"}


def test_src_change_flags_direct_dependent_app_via_path(tmp_path: Path) -> None:
    write_pyproject(tmp_path / "src/budget/pyproject.toml")
    write_pyproject(
        tmp_path / "apps/tools-app/budget-app/pyproject.toml",
        sources='budget = { path = "../../../src/budget", editable = true }',
    )
    write_pyproject(tmp_path / "apps/unrelated/pyproject.toml")

    affected = compute_affected_apps(
        tmp_path,
        apps_files=[],
        src_files=["src/budget/wiz/budget/schemas.py"],
    )

    assert affected == {"tools-app/budget-app"}


def test_src_change_flags_app_transitively_via_workspace_dependency(tmp_path: Path) -> None:
    # src/shared depends on src/evaluation via workspace=true; house-prices depends on
    # src/shared via path=. A src/evaluation change must reach house-prices even though
    # house-prices never references evaluation directly.
    write_pyproject(tmp_path / "src/evaluation/pyproject.toml")
    write_pyproject(
        tmp_path / "src/shared/pyproject.toml",
        sources="evaluation = { workspace = true }",
    )
    write_pyproject(
        tmp_path / "apps/house-prices/pyproject.toml",
        sources='shared = { path = "../../src/shared", editable = true }',
    )

    affected = compute_affected_apps(
        tmp_path,
        apps_files=[],
        src_files=["src/evaluation/wiz/evaluation/metrics.py"],
    )

    assert affected == {"house-prices"}


def test_non_src_path_dependency_is_ignored(tmp_path: Path) -> None:
    # tools-app depends on its own power-app subdirectory by path -- that's an app-to-app
    # dependency, not a src/* package, and must not be treated as one.
    write_pyproject(
        tmp_path / "apps/tools-app/pyproject.toml",
        sources='power-app = { path = "./power-app", editable = true }',
    )
    write_pyproject(tmp_path / "apps/tools-app/power-app/pyproject.toml")

    affected = compute_affected_apps(
        tmp_path,
        apps_files=[],
        src_files=["src/power-app/some_file.py"],
    )

    assert affected == set()


def test_ci_workflow_change_runs_every_app(tmp_path: Path) -> None:
    write_pyproject(tmp_path / "apps/finance/pyproject.toml")
    write_pyproject(tmp_path / "apps/tools-app/budget-app/pyproject.toml")

    affected = compute_affected_apps(
        tmp_path,
        apps_files=[CI_WORKFLOW_PATH],
        src_files=[],
    )

    assert affected == {"finance", "tools-app/budget-app"}


def test_against_real_repo_matches_issue_16_scenario() -> None:
    """Regression test for the exact scenario reported in issue #16: a change confined to
    src/budget must flag apps/tools-app/budget-app (which imports wiz.budget)."""
    affected = compute_affected_apps(
        REPO_ROOT,
        apps_files=[],
        src_files=["src/budget/wiz/budget/schemas.py"],
    )

    assert "tools-app/budget-app" in affected
    assert "house-prices" not in affected

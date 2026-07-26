import io
import sys
from dataclasses import dataclass

import pytest

from pants_addrs_to_apps import main, resolve_app_dir


@dataclass
class ResolveCase:
    id: str
    address: str
    expected: str | None


RESOLVE_CASES = [
    ResolveCase(
        id="bare_generator_address_is_the_app_dir",
        address="apps/churn:lib",
        expected="churn",
    ),
    ResolveCase(
        id="file_at_apps_top_level",
        address="apps/churn/helper_functions.py:lib",
        expected="churn",
    ),
    ResolveCase(
        id="file_one_level_nested",
        address="apps/finance/src/dataloader.py:../lib",
        expected="finance",
    ),
    ResolveCase(
        id="file_two_levels_nested",
        address="apps/weather-data-db/src/loaders/dmi_client_wrapper.py:../../lib",
        expected="weather-data-db",
    ),
    ResolveCase(
        id="nested_app_itself_is_the_app_dir",
        address="apps/tools-app/budget-app:lib",
        expected="tools-app/budget-app",
    ),
    ResolveCase(
        id="file_nested_inside_a_nested_app",
        address="apps/tools-app/power-app/pages/overview.py:../lib",
        expected="tools-app/power-app",
    ),
    ResolveCase(
        # Regression test: the "../" count must be read structurally, not by stripping a
        # hardcoded "lib" suffix -- a BUILD file naming its target anything else must still work.
        id="generator_name_other_than_lib_does_not_break_parsing",
        address="apps/weather-data-db/src/loaders/x.py:../../sources",
        expected="weather-data-db",
    ),
    ResolveCase(
        id="generator_name_with_no_up_levels_and_a_different_name",
        address="apps/churn/helper_functions.py:py_sources",
        expected="churn",
    ),
    ResolveCase(
        id="non_apps_address_is_ignored",
        address="src/budget/wiz/budget/schemas.py:../../lib",
        expected=None,
    ),
    ResolveCase(
        id="malformed_address_without_colon_is_ignored",
        address="not-an-address",
        expected=None,
    ),
]


@pytest.mark.parametrize("case", RESOLVE_CASES, ids=lambda case: case.id)
def test_resolve_app_dir(case: ResolveCase) -> None:
    assert resolve_app_dir(case.address) == case.expected


def test_main_dedupes_and_sorts_output(monkeypatch, capsys) -> None:
    addresses = "\n".join(
        [
            "apps/finance:lib",
            "apps/finance/app.py:lib",
            "apps/churn/helper_functions.py:lib",
            "",
            "src/budget/wiz/budget/schemas.py:../../lib",
        ]
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(addresses))

    main()

    assert capsys.readouterr().out == "churn\nfinance\n"

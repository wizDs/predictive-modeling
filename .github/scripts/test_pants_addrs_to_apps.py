import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pants_addrs_to_apps import main, resolve_app_dir


def test_bare_generator_address_is_the_app_dir() -> None:
    assert resolve_app_dir("apps/churn:lib") == "churn"


def test_file_at_apps_top_level() -> None:
    assert resolve_app_dir("apps/churn/helper_functions.py:lib") == "churn"


def test_file_one_level_nested() -> None:
    assert resolve_app_dir("apps/finance/src/dataloader.py:../lib") == "finance"


def test_file_two_levels_nested() -> None:
    address = "apps/weather-data-db/src/loaders/dmi_client_wrapper.py:../../lib"
    assert resolve_app_dir(address) == "weather-data-db"


def test_nested_app_itself_is_the_app_dir() -> None:
    assert resolve_app_dir("apps/tools-app/budget-app:lib") == "tools-app/budget-app"


def test_file_nested_inside_a_nested_app() -> None:
    address = "apps/tools-app/power-app/pages/overview.py:../lib"
    assert resolve_app_dir(address) == "tools-app/power-app"


def test_generator_name_other_than_lib_does_not_break_parsing() -> None:
    # Regression test: the "../" count must be read structurally, not by stripping a
    # hardcoded "lib" suffix -- a BUILD file naming its target anything else must still work.
    address = "apps/weather-data-db/src/loaders/x.py:../../sources"
    assert resolve_app_dir(address) == "weather-data-db"


def test_generator_name_with_no_up_levels_and_a_different_name() -> None:
    assert resolve_app_dir("apps/churn/helper_functions.py:py_sources") == "churn"


def test_non_apps_address_is_ignored() -> None:
    assert resolve_app_dir("src/budget/wiz/budget/schemas.py:../../lib") is None


def test_malformed_address_without_colon_is_ignored() -> None:
    assert resolve_app_dir("not-an-address") is None


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

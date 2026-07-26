#!/usr/bin/env python3
"""Map `pants list` target addresses (stdin, one per line) to their owning apps/<dir>, for
ci.yml's `changes` job. Prints each affected app dir once, sorted.

An address is either a bare generator (apps/<dir>:name -- the path is the app dir) or a
generated file target (apps/<dir>/<nested>/file.py:<../..>name), where the "../" count in the
suffix is how many directories to walk up from the file to reach the app dir. That count is
read structurally via regex, not by stripping a hardcoded generator name like "lib", since
nothing guarantees every BUILD file names its target that.
"""

import re
import sys
from pathlib import PurePosixPath

UP_LEVELS_RE = re.compile(r"^(\.\./)*")


def resolve_app_dir(address: str) -> str | None:
    path, sep, name_suffix = address.partition(":")
    if not sep or not path.startswith("apps/"):
        return None

    if not path.endswith(".py"):
        # A bare generator address, e.g. apps/churn:lib -- the path *is* the app dir.
        return path.removeprefix("apps/")

    up_levels = len(UP_LEVELS_RE.match(name_suffix).group(0)) // len("../")
    directory = PurePosixPath(path).parent
    for _ in range(up_levels):
        directory = directory.parent
    return str(directory).removeprefix("apps/")


def main() -> None:
    apps = set()
    for line in sys.stdin:
        address = line.strip()
        if not address:
            continue
        app_dir = resolve_app_dir(address)
        if app_dir:
            apps.add(app_dir)
    for app_dir in sorted(apps):
        print(app_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Map `pants list` target addresses (read one per line from stdin) to their owning apps/<dir>,
for ci.yml's `changes` job. Prints each affected app dir once, sorted, one per line.

An address is either a generator itself (apps/<dir>:<name>) or a generated file target
(apps/<dir>/<nested>/<file>.py:<up><name>, e.g. apps/weather-data-db/src/loaders/x.py:../../lib)
-- the target-name part after the colon is some number of "../" (however many directories the
file sits below its app's own directory) followed by the generator's own name. Walking up that
many directories from the file reaches the owning app dir.

The "../" count is read structurally (a regex match on the leading "../" run), not by stripping
a hardcoded generator name like "lib" off the end -- every apps/*/BUILD in this repo happens to
name its target "lib" today, but nothing enforces that, and string-stripping a literal "lib"
would silently produce a wrong path the moment a BUILD file names its target anything else.
"""

from __future__ import annotations

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

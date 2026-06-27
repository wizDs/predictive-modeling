# Import the py_library rule (defines a Python library target) from rules_python.
load("@rules_python//python:defs.bzl", "py_library")
# Import requirement() from the @pip hub — maps a PyPI package name to its Bazel target.
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "budget",  # Target label used by dependents: //src/budget:budget
    # glob() collects all .py files under wiz/budget/; Bazel tracks them for incremental rebuilds.
    srcs = glob(["wiz/budget/**/*.py"]),
    # Adds src/budget/ to sys.path so `import wiz.budget` resolves correctly at build time.
    imports = ["."],
    # Explicit dep list — Bazel enforces that only packages listed here can be imported.
    # Anything missing here causes a build error, catching undeclared deps early.
    deps = [
        requirement("pydantic"),
        requirement("polars"),
        requirement("pyarrow"),
        requirement("pandas"),
        requirement("python-dateutil"),
        requirement("requests"),
        requirement("google-auth-oauthlib"),
        requirement("google-auth-httplib2"),
        requirement("google-api-python-client"),
    ],
    # Makes this target importable by other packages in the repo (e.g. apps that depend on budget).
    visibility = ["//visibility:public"],
)

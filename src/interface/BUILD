# Import the py_library rule (defines a Python library target) from rules_python.
load("@rules_python//python:defs.bzl", "py_library")
# Import requirement() from the @pip hub — maps a PyPI package name to its Bazel target.
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "interface",  # Target label used by dependents: //src/interface:interface
    # glob() picks up all .py files including feature_array.py automatically as new files are added.
    srcs = glob(["wiz/interface/**/*.py"]),
    # Adds src/interface/ to sys.path so `import wiz.interface` resolves.
    imports = ["."],
    deps = [
        requirement("pydantic"),
        requirement("xgboost"),
        requirement("numpy"),
    ],
    # Makes this target importable by shared, evaluation, and any app that needs the interfaces.
    visibility = ["//visibility:public"],
)

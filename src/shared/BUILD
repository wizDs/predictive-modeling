# Import the py_library rule (defines a Python library target) from rules_python.
load("@rules_python//python:defs.bzl", "py_library")
# Import requirement() from the @pip hub — maps a PyPI package name to its Bazel target.
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "shared",  # Target label used by dependents: //src/shared:shared
    # glob() collects all .py files recursively, including subdirectories like estimator/.
    srcs = glob(["wiz/shared/**/*.py"]),
    # Adds src/shared/ to sys.path so `import wiz.shared` resolves.
    imports = ["."],
    deps = [
        # Depends on evaluation_core (metric.py only), NOT :evaluation.
        # helpers.py in evaluation imports wiz.shared, so depending on the full :evaluation
        # target here would create a circular dependency that Bazel rejects.
        "//src/evaluation:evaluation_core",
        "//src/interface:interface",  # Absolute label for the interface package.
        requirement("scikit-learn"),
        requirement("numpy"),
        requirement("xgboost"),
        requirement("lightgbm"),
        requirement("polars"),
        requirement("pyarrow"),
        requirement("pandas"),
        requirement("toolz"),
    ],
    # Makes this target importable by apps and the full :evaluation target.
    visibility = ["//visibility:public"],
)

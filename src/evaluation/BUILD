# Import the py_library rule (defines a Python library target) from rules_python.
load("@rules_python//python:defs.bzl", "py_library")
# Import requirement() from the @pip hub — maps a PyPI package name to its Bazel target.
load("@pip//:requirements.bzl", "requirement")

# evaluation_core contains only metric.py — no intra-wiz imports.
# It is split out so that //src/shared can depend on just the metrics
# without creating a circular dependency (shared <- evaluation <- shared).
py_library(
    name = "evaluation_core",  # Target label: //src/evaluation:evaluation_core
    srcs = [
        "wiz/evaluation/__init__.py",
        "wiz/evaluation/metric.py",
    ],
    # Adds src/evaluation/ to sys.path so `import wiz.evaluation` resolves.
    imports = ["."],
    deps = [
        requirement("scikit-learn"),
        requirement("numpy"),
    ],
    visibility = ["//visibility:public"],
)

# Full evaluation target: adds helpers which import wiz.shared and wiz.interface.
# Depends on shared, so shared must only reference :evaluation_core (not this target).
py_library(
    name = "evaluation",  # Target label: //src/evaluation:evaluation
    srcs = [
        "wiz/evaluation/helpers.py",
        "wiz/evaluation/helpers2.py",
    ],
    imports = ["."],
    deps = [
        ":evaluation_core",         # Reuse the core metrics target defined above.
        "//src/shared:shared",      # Absolute label — references the shared package target.
        "//src/interface:interface",
        requirement("scikit-learn"),
        requirement("numpy"),
        requirement("polars"),
        requirement("pydantic"),
    ],
    visibility = ["//visibility:public"],
)

load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "evaluation_core",
    srcs = [
        "wiz/evaluation/__init__.py",
        "wiz/evaluation/metric.py",
    ],
    imports = ["."],
    deps = [
        requirement("scikit-learn"),
        requirement("numpy"),
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "evaluation",
    srcs = [
        "wiz/evaluation/helpers.py",
        "wiz/evaluation/helpers2.py",
    ],
    imports = ["."],
    deps = [
        ":evaluation_core",
        "//src/shared:shared",
        "//src/interface:interface",
        requirement("scikit-learn"),
        requirement("numpy"),
        requirement("polars"),
        requirement("pydantic"),
    ],
    visibility = ["//visibility:public"],
)

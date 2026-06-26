load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "shared",
    srcs = glob(["wiz/shared/**/*.py"]),
    imports = ["."],
    deps = [
        "//src/evaluation:evaluation_core",
        "//src/interface:interface",
        requirement("scikit-learn"),
        requirement("numpy"),
        requirement("xgboost"),
        requirement("lightgbm"),
        requirement("polars"),
        requirement("pyarrow"),
        requirement("pandas"),
        requirement("toolz"),
    ],
    visibility = ["//visibility:public"],
)

load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "interface",
    srcs = glob(["wiz/interface/**/*.py"]),
    imports = ["."],
    deps = [
        requirement("pydantic"),
        requirement("xgboost"),
        requirement("numpy"),
    ],
    visibility = ["//visibility:public"],
)

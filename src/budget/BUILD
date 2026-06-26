load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "budget",
    srcs = glob(["wiz/budget/**/*.py"]),
    imports = ["."],
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
    visibility = ["//visibility:public"],
)

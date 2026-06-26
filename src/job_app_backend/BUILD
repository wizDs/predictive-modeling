load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "job_app_backend",
    srcs = glob(["wiz/job_app_backend/**/*.py"]),
    imports = ["."],
    deps = [requirement("spacy")],
    visibility = ["//visibility:public"],
)

# Import the py_library rule (defines a Python library target) from rules_python.
load("@rules_python//python:defs.bzl", "py_library")
# Import requirement() from the @pip hub — maps a PyPI package name to its Bazel target.
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "job_app_backend",  # Target label: //src/job_app_backend:job_app_backend
    # glob() collects all .py files; NER model data files are not .py so are excluded automatically.
    srcs = glob(["wiz/job_app_backend/**/*.py"]),
    # Adds src/job_app_backend/ to sys.path so `import wiz.job_app_backend` resolves.
    imports = ["."],
    deps = [requirement("spacy")],
    visibility = ["//visibility:public"],
)

python_requirements(name="reqs")

# Mirrors the root pyproject.toml's [dependency-groups] dev list. Pants' python_requirements()
# target only reads [project.dependencies]/[project.optional-dependencies], not PEP 735
# dependency-groups, so this can't be generated from that file -- keep them in sync by hand.
python_requirement(
    name="mypy-tool-reqs",
    requirements=[
        "mypy>=1.15",
        "pandas-stubs>=2.2",
        "types-python-dateutil>=2.9",
        "types-requests>=2.32",
    ],
    resolve="mypy-tool",
)

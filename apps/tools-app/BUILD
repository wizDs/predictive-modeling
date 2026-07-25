# budget-app, power-app, and transcribe are separate source roots with their own BUILD files
# (apps/tools-app/*/BUILD) -- excluded here so a file isn't claimed by two targets. Mirrors the
# equivalent exclude in this app's own pyproject.toml [tool.mypy] config.
python_sources(
    name="lib",
    tags=["app"],
    sources=["**/*.py", "!budget-app/**", "!power-app/**", "!transcribe/**"],
)

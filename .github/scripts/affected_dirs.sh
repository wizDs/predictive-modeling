#!/usr/bin/env bash
# Prints the space-separated set of src/* package directories affected by the diff against
# $1 (default origin/main), using Bazel's dependency graph to include transitive dependents.
# Prints nothing (and exits 0) if nothing relevant changed.
set -euo pipefail

base_ref="${1:-origin/main}"

# Fail loudly rather than silently treating an unresolvable ref as "nothing changed" -- a
# missing base_ref (e.g. a checkout that didn't fetch it) must not be able to make CI silently
# skip every affected package.
if ! git rev-parse --verify --quiet "${base_ref}" > /dev/null; then
  echo "error: base ref '${base_ref}' does not resolve; check the checkout's fetch-depth" >&2
  exit 1
fi

# A change to MODULE.bazel or the root BUILD.bazel can affect the whole graph (toolchain
# version, root package config) in ways rdeps can't express -- bail out to a full check.
if ! git diff --quiet "${base_ref}...HEAD" -- MODULE.bazel BUILD.bazel; then
  echo "MODULE.bazel/BUILD.bazel changed; falling back to a full check." >&2
  bazel query "kind(py_library, //src/...)" 2>/dev/null \
    | sed -E 's#^//([^:]+):.*#\1#' \
    | sort -u \
    | tr '\n' ' '
  exit 0
fi

changed_py_files=$(git diff --name-only "${base_ref}...HEAD" -- 'src/*.py' 'src/**/*.py')

# A src/*/BUILD.bazel change (new/changed deps, new target) invalidates that package's targets
# even with no .py changes -- query rdeps on the package itself (//src/foo/...), not the
# BUILD.bazel file path, since rdeps doesn't treat BUILD files as graph nodes.
changed_build_dirs=$(git diff --name-only "${base_ref}...HEAD" -- 'src/*/BUILD.bazel' \
  | sed -E 's#/BUILD\.bazel$##')

if [ -z "${changed_py_files}" ] && [ -z "${changed_build_dirs}" ]; then
  exit 0
fi

query_set="${changed_py_files}"
for dir in ${changed_build_dirs}; do
  query_set="${query_set} //${dir}/..."
done
query_set=$(echo "${query_set}" | tr '\n' ' ')

bazel query "kind(py_library, rdeps(//src/..., set(${query_set})))" 2>/dev/null \
  | sed -E 's#^//([^:]+):.*#\1#' \
  | sort -u \
  | tr '\n' ' '

#!/usr/bin/env bash
# check_stage_paths.sh - refuse to stage any path covered by .gitignore.
#
# Why this exists: from 2026-04-29 to 2026-07-17 the threat-feed
# refresh workflow failed every scheduled run (91 in a row) because
# its commit step staged data/feeds/, which .gitignore excludes.
# `git add` exits 1 when a pathspec matches only ignored files, the
# step aborted before committing, and the failure went unnoticed
# for three months.
#
# Usage: check_stage_paths.sh <path> [<path> ...]
# Exit 0: every path is committable. Exit 1: at least one path is
# covered by .gitignore. The check runs on the exact pathspecs the
# caller is about to stage, using git's own ignore semantics.
#
# Scope, stated plainly: this checks the staging pathspecs
# themselves. It does not inspect every file nested inside an
# allowed directory. That matches the proven failure class (a
# wholly ignored pathspec aborts `git add`); a partially ignored
# directory does not abort `git add`, because git stages its
# non-ignored files. Do not describe this script as proving every
# possible staging path safe.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: check_stage_paths.sh <path> [<path> ...]" >&2
  exit 2
fi

for p in "$@"; do
  if git check-ignore -q -- "$p"; then
    echo "::error::Staging path '$p' is covered by .gitignore. Refusing to stage it." >&2
    exit 1
  fi
done

echo "All staging paths are committable: $*"

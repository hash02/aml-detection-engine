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

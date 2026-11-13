#!/usr/bin/env bash
# Checks a branch name against the naming scheme in HANDOFF.md ("Branch names").
# Usage: scripts/check-branch-name.sh [branch]   (defaults to the current branch)
set -euo pipefail

branch="${1:-$(git rev-parse --abbrev-ref HEAD)}"

# Long-lived or tool-made branches that keep their own names.
case "$branch" in
  main|noticeboard|claude/*|dependabot/*) exit 0 ;;
esac

types='feat|fix|docs|refactor|test|ml|ci|chore'
pattern="^($types)/[a-z0-9]+(-[a-z0-9]+)*$"
max=50

if [[ ! "$branch" =~ $pattern ]]; then
  cat >&2 <<MSG
Branch name "$branch" does not follow the naming scheme.

  Use:      <type>/<short-description-in-kebab-case>
  Types:    ${types//|/, }
  Examples: feat/practice-streaks, fix/ws-origin-check, ml/temporal-conv-ablation

Rename it with:  git branch -m <new-name>
MSG
  exit 1
fi

if (( ${#branch} > max )); then
  echo "Branch name \"$branch\" is ${#branch} characters; keep it to $max or fewer." >&2
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

# Auto-stage, commit, and push changes for the current repository.
# Usage:
#   ./scripts/auto_commit_push.sh "your commit message"
#   ./scripts/auto_commit_push.sh "your commit message" origin

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 \"commit message\" [remote]"
  exit 1
fi

commit_message="$1"
remote="${2:-origin}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: this script must run inside a git repository."
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" == "HEAD" ]]; then
  echo "Error: detached HEAD state detected. Checkout a branch first."
  exit 1
fi

# Stage all tracked and untracked changes.
git add -A

# Exit early if there is nothing staged.
if git diff --cached --quiet; then
  echo "No staged changes to commit."
  exit 0
fi

git commit -m "$commit_message"
git push "$remote" "$branch"

echo "Done: committed and pushed to $remote/$branch"

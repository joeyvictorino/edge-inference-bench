#!/usr/bin/env bash
# Fail if any tracked or working-tree text file names an organisation or
# product that must not appear in this repository.
#
# The pattern itself is not stored in the repository, because listing the
# names here would publish them. It is read from, in order:
#   1. the NAME_AUDIT_PATTERN environment variable (a repository secret in CI)
#   2. ${XDG_CONFIG_HOME:-$HOME/.config}/name-audit/pattern (one extended regex)
# With neither, the audit fails (exit 2), unless NAME_AUDIT_ALLOW_MISSING=1,
# which CI sets only for pull requests from forks, where secrets are absent.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PATTERN="${NAME_AUDIT_PATTERN:-}"
CONF="${XDG_CONFIG_HOME:-$HOME/.config}/name-audit/pattern"
if [ -z "$PATTERN" ] && [ -r "$CONF" ]; then
  PATTERN=$(head -n 1 "$CONF")
fi
if [ -z "$PATTERN" ]; then
  if [ "${NAME_AUDIT_ALLOW_MISSING:-0}" = "1" ]; then
    echo "name audit skipped (no pattern configured; allowed for this run)"
    exit 0
  fi
  echo "name audit ERROR: set NAME_AUDIT_PATTERN or create $CONF" >&2
  exit 2
fi
if hits=$(grep -rnilE --binary-files=without-match \
    --exclude-dir=.git --exclude-dir=models --exclude-dir=.venv --exclude-dir=__pycache__ \
    --exclude-dir=.hf-cache --exclude=name-audit.sh \
    -- "$PATTERN" .); then
  # File names only, never the matched text, so CI logs do not echo it.
  echo "name audit FAILED; forbidden names found in:" >&2
  echo "$hits" >&2
  exit 1
fi
echo "name audit passed"

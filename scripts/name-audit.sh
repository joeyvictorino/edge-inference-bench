#!/usr/bin/env bash
# Fail if any tracked or working-tree text file names an organisation or
# product that must not appear in this repository. The pattern lives in
# this file only, so the file audits everything except itself.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PATTERN='qompute|qomputeai|\bqore\b|palo alto|unit 42|requisition'
if hits=$(grep -rniE --binary-files=without-match \
    --exclude-dir=.git --exclude-dir=models --exclude-dir=.venv --exclude-dir=__pycache__ \
    --exclude-dir=.hf-cache --exclude=name-audit.sh \
    -- "$PATTERN" .); then
  echo "name audit FAILED; forbidden names found:" >&2
  echo "$hits" >&2
  exit 1
fi
echo "name audit passed"

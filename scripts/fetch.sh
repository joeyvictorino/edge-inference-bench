#!/usr/bin/env bash
# Download one GGUF listed in models.yaml into models/ and record its
# sha256 and byte size in models.lock.
#
#   scripts/fetch.sh MODEL_ID
#
# Resumable (curl -C -). Idempotent: if the file is present and its digest
# already matches models.lock, nothing is downloaded or rewritten.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ID="${1:-}"
if [ -z "$ID" ]; then
  echo "usage: $0 MODEL_ID   (ids: $(python3 "$ROOT/scripts/lib/modelsyaml.py" ids | tr '\n' ' '))" >&2
  exit 2
fi

REPO=$(python3 "$ROOT/scripts/lib/modelsyaml.py" get "$ID" repo)
FILE=$(python3 "$ROOT/scripts/lib/modelsyaml.py" get "$ID" file)
[ -n "$REPO" ] && [ -n "$FILE" ] || { echo "model id not found in models.yaml: $ID" >&2; exit 1; }

URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"
DEST_DIR="$ROOT/models"
DEST="$DEST_DIR/$FILE"
LOCK="$ROOT/models.lock"
mkdir -p "$DEST_DIR"
[ -f "$LOCK" ] || echo '{}' > "$LOCK"

locked_sha=$(python3 - "$LOCK" "$ID" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get(sys.argv[2], {}).get("sha256", ""))
PY
)

if [ -f "$DEST" ] && [ -n "$locked_sha" ]; then
  actual=$(shasum -a 256 "$DEST" | awk '{print $1}')
  if [ "$actual" = "$locked_sha" ]; then
    echo "ok: $FILE already present, sha256 matches models.lock"
    exit 0
  fi
  echo "warn: $FILE present but sha256 differs from models.lock; re-downloading"
fi

echo "fetching $URL"
AUTH=()
if [ -n "${HF_TOKEN:-}" ]; then AUTH=(-H "Authorization: Bearer $HF_TOKEN"); fi
# -C - resumes a partial file; --fail makes HTTP errors fatal; -L follows the LFS redirect.
curl -L --fail --retry 5 --retry-delay 3 -C - ${AUTH[@]+"${AUTH[@]}"} -o "$DEST" "$URL"

SHA=$(shasum -a 256 "$DEST" | awk '{print $1}')
SIZE=$(stat -f %z "$DEST")

python3 - "$LOCK" "$ID" "$REPO" "$FILE" "$SHA" "$SIZE" <<'PY'
import json, sys
lock_path, mid, repo, fname, sha, size = sys.argv[1:7]
lock = json.load(open(lock_path))
lock[mid] = {
    "repo": repo,
    "file": fname,
    "url": "https://huggingface.co/%s/resolve/main/%s" % (repo, fname),
    "sha256": sha,
    "size_bytes": int(size),
}
with open(lock_path, "w") as fh:
    json.dump(lock, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY

echo "ok: $FILE sha256=$SHA size=$SIZE -> models.lock"

#!/usr/bin/env bash
# Throughput versus prompt length for one GGUF.
#
#   scripts/context_length.sh MODEL_PATH OUT_DIR
#
# One llama-bench call per prompt size so that an out-of-memory failure at a
# long context does not lose the shorter points. Defaults:
#   -p 512 1024 2048 4096 8192 16384   -n 64   -r 3
#   threads: llama-bench default unless THREADS is set (see sweep for tuning)
#   other options: llama-bench defaults (-fa auto, f16 KV cache)
#
# Resumable: existing p<N>.json files are skipped. Failures are recorded as
# p<N>.failed.json. Environment: PROMPTS, GEN, REPS, THREADS, SMOKE=1.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=lib/host.sh
source "$ROOT/scripts/lib/host.sh"

MODEL="${1:-}"; OUT="${2:-}"
if [ -z "$MODEL" ] || [ -z "$OUT" ]; then
  echo "usage: $0 MODEL_PATH OUT_DIR" >&2; exit 2
fi
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
command -v llama-bench >/dev/null || { echo "llama-bench not on PATH" >&2; exit 1; }

PROMPTS="${PROMPTS:-512 1024 2048 4096 8192 16384}"
GEN="${GEN:-64}"
REPS="${REPS:-3}"
THREAD_ARGS=()
if [ -n "${THREADS:-}" ]; then THREAD_ARGS=(-t "$THREADS"); fi
if [ "${SMOKE:-0}" = "1" ]; then PROMPTS="128"; GEN=16; REPS=1; fi

mkdir -p "$OUT"
TMP="$OUT/raw-tmp"; mkdir -p "$TMP"

write_manifest "$OUT" "$MODEL" "context_length" "$(jq -n \
  --arg prompts "$PROMPTS" --arg gen "$GEN" --arg reps "$REPS" --arg threads "${THREADS:-default}" \
  --argjson smoke "$([ "${SMOKE:-0}" = "1" ] && echo true || echo false)" \
  '{grid:{prompts:$prompts, gen:($gen|tonumber), repetitions:($reps|tonumber), threads:$threads}, smoke:$smoke}')"

for p in $PROMPTS; do
  name="p${p}"
  final="$OUT/$name.json"
  if [ -s "$final" ]; then echo "[$name] exists, skipping"; continue; fi
  if [ -s "$OUT/$name.failed.json" ] && [ "${RETRY_FAILED:-0}" != "1" ]; then continue; fi
  echo "[$name] -p $p -n $GEN -r $REPS ${THREAD_ARGS[*]:-}"
  set +e
  llama-bench -m "$MODEL" ${THREAD_ARGS[@]+"${THREAD_ARGS[@]}"} -p "$p" -n "$GEN" -r "$REPS" -o json \
    > "$TMP/$name.json" 2> "$TMP/$name.stderr"
  rc=$?
  set -e
  if [ $rc -eq 0 ] && jq -e 'type=="array" and length>0' "$TMP/$name.json" >/dev/null 2>&1; then
    mv "$TMP/$name.json" "$final"; rm -f "$TMP/$name.stderr" "$OUT/$name.failed.json"
  else
    jq -n --arg name "$name" --argjson rc "$rc" \
      --arg stderr "$(tail -c 4000 "$TMP/$name.stderr" 2>/dev/null || true)" \
      '{config:$name, exit_code:$rc, stderr_tail:$stderr}' > "$OUT/$name.failed.json"
    rm -f "$TMP/$name.json" "$TMP/$name.stderr"
    echo "  failed (exit $rc); recorded $name.failed.json"
  fi
done
rmdir "$TMP" 2>/dev/null || true
echo "context_length complete -> $OUT"

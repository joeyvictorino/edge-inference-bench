#!/usr/bin/env bash
# Run llama-bench over the configuration grid for one GGUF and write one raw
# JSON file per configuration into OUT_DIR, plus OUT_DIR/manifest.json.
#
#   scripts/sweep.sh MODEL_PATH OUT_DIR
#
# Grid (full run):
#   threads      2 4 6
#   batch/ubatch 256:{128,256} 512:{128,256,512} 1024:{128,256,512}   (ubatch <= batch)
#   flash-attn   off on            (-fa off|on; files are named fa0 / fa1)
#   KV cache     f16 q8_0          (applied to both -ctk and -ctv)
#   prompt       512 2048 8192     (-p, all in one llama-bench call)
#   generation   128               (-n)
#   repetitions  3                 (-r)
#
# Resumable: a configuration whose JSON already exists is skipped. A
# configuration that llama-bench cannot run (for example a quantised V cache
# with flash attention off, or an out-of-memory 8192-token prompt) is recorded
# as <name>.failed.json with the exit code and the tail of stderr, and the
# sweep continues. summarize.py counts failed configurations but never
# reports numbers from them.
#
# Measurement conditions: refuses to run on battery (ALLOW_BATTERY=1 to
# override; such runs are recorded and excluded from summaries), kills a
# configuration that runs longer than CONFIG_TIMEOUT seconds (default 1800),
# and records <name>.env.json beside every result. A configuration during
# which the machine slept is kept as <name>.interrupted.json, never counted,
# and retried on the next run.
#
# Environment overrides:
#   SMOKE=1      one configuration, -p 128 -n 32 -r 1 (pipeline check only)
#   REPS=n       repetitions (default 3)
#   PROMPTS=a,b  prompt sizes (default 512,2048,8192)
#   GEN=n        generation length (default 128)
#   THREADS, BATCHES ("b:ub b:ub ..."), FA ("off on"), KV ("f16 q8_0")
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
require_ac_power
CONFIG_TIMEOUT="${CONFIG_TIMEOUT:-1800}"   # seconds; a configuration still running after this is killed and recorded

REPS="${REPS:-3}"
PROMPTS="${PROMPTS:-512,2048,8192}"
GEN="${GEN:-128}"
THREADS="${THREADS:-2 4 6}"
BATCHES="${BATCHES:-256:128 256:256 512:128 512:256 512:512 1024:128 1024:256 1024:512}"
FA="${FA:-off on}"
KV="${KV:-f16 q8_0}"

if [ "${SMOKE:-0}" = "1" ]; then
  REPS=1; PROMPTS=128; GEN=32
  THREADS=4; BATCHES="512:256"; FA="on"; KV="f16"
  echo "SMOKE=1: single configuration, -p $PROMPTS -n $GEN -r $REPS"
fi

mkdir -p "$OUT"
TMP="$OUT/raw-tmp"; mkdir -p "$TMP"

write_manifest "$OUT" "$MODEL" "sweep" "$(jq -n \
  --arg reps "$REPS" --arg prompts "$PROMPTS" --arg gen "$GEN" \
  --arg threads "$THREADS" --arg batches "$BATCHES" --arg fa "$FA" --arg kv "$KV" \
  --argjson smoke "$([ "${SMOKE:-0}" = "1" ] && echo true || echo false)" \
  '{grid:{repetitions:($reps|tonumber), prompts:$prompts, gen:($gen|tonumber),
          threads:$threads, batches:$batches, flash_attn:$fa, kv_cache:$kv}, smoke:$smoke}')"

total=0; done_n=0; skipped=0; failed=0; interrupted=0
for t in $THREADS; do
  for bu in $BATCHES; do
    b="${bu%%:*}"; ub="${bu##*:}"
    [ "$ub" -le "$b" ] || continue
    for fa in $FA; do
      fa_tag=$([ "$fa" = "on" ] && echo 1 || echo 0)
      for kv in $KV; do
        total=$((total + 1))
        name="t${t}_b${b}_ub${ub}_fa${fa_tag}_kv${kv}"
        final="$OUT/$name.json"
        if [ -s "$final" ]; then skipped=$((skipped + 1)); continue; fi
        if [ -s "$OUT/$name.failed.json" ] && [ "${RETRY_FAILED:-0}" != "1" ]; then
          skipped=$((skipped + 1)); continue
        fi
        echo "[$name] -p $PROMPTS -n $GEN -r $REPS"
        power_before=$(power_json); sleep_before=$(sleep_stamp); started=$(date +%s)
        set +e
        llama-bench -m "$MODEL" -t "$t" -b "$b" -ub "$ub" -fa "$fa" \
          -ctk "$kv" -ctv "$kv" -p "$PROMPTS" -n "$GEN" -r "$REPS" \
          -o json > "$TMP/$name.json" 2> "$TMP/$name.stderr" &
        bench_pid=$!
        # Watchdog: polls once a second and ends by itself when the benchmark
        # does, so it never leaves a long sleep behind holding our output open.
        ( i=0
          while [ "$i" -lt "$CONFIG_TIMEOUT" ] && kill -0 "$bench_pid" 2>/dev/null; do
            sleep 1; i=$((i + 1))
          done
          kill "$bench_pid" 2>/dev/null ) &
        watchdog_pid=$!
        wait "$bench_pid"
        rc=$?
        wait "$watchdog_pid" 2>/dev/null
        set -e
        ended=$(date +%s)
        power_after=$(power_json); sleep_after=$(sleep_stamp)
        slept=false; [ "$sleep_before" = "$sleep_after" ] || slept=true
        timed_out=false; [ $rc -eq 143 ] && [ $((ended - started)) -ge "$CONFIG_TIMEOUT" ] && timed_out=true
        # Conditions record: lets summarize.py exclude runs that were not clean.
        jq -n --argjson before "$power_before" --argjson after "$power_after" \
          --argjson slept "$slept" --argjson timed_out "$timed_out" \
          --argjson seconds "$((ended - started))" \
          '{power_before:$before, power_after:$after, slept_during_run:$slept,
            timed_out:$timed_out, wall_seconds:$seconds}' > "$OUT/$name.env.json"
        if [ "$slept" = true ]; then
          # The machine slept while this ran, so its timings include the sleep.
          # Keep the raw output for diagnosis but never count it; the next run retries it.
          [ -s "$TMP/$name.json" ] && mv "$TMP/$name.json" "$OUT/$name.interrupted.json"
          rm -f "$TMP/$name.json" "$TMP/$name.stderr"
          interrupted=$((interrupted + 1))
          echo "  interrupted by system sleep; recorded $name.interrupted.json (will be retried)"
        elif [ $rc -eq 0 ] && jq -e 'type=="array" and length>0' "$TMP/$name.json" >/dev/null 2>&1; then
          # keep the raw llama-bench array; add nothing (manifest carries the host data)
          mv "$TMP/$name.json" "$final"
          rm -f "$TMP/$name.stderr" "$OUT/$name.failed.json" "$OUT/$name.interrupted.json"
          done_n=$((done_n + 1))
        else
          failed=$((failed + 1))
          jq -n --arg name "$name" --argjson rc "$rc" \
            --arg stderr "$(tail -c 4000 "$TMP/$name.stderr" 2>/dev/null || true)" \
            --arg cmd "llama-bench -t $t -b $b -ub $ub -fa $fa -ctk $kv -ctv $kv -p $PROMPTS -n $GEN -r $REPS" \
            '{config:$name, exit_code:$rc, command:$cmd, stderr_tail:$stderr}' > "$OUT/$name.failed.json"
          rm -f "$TMP/$name.json" "$TMP/$name.stderr"
          echo "  failed (exit $rc); recorded $name.failed.json"
        fi
      done
    done
  done
done
rmdir "$TMP" 2>/dev/null || true
echo "sweep complete: $total configs, $done_n new, $skipped skipped, $failed failed, $interrupted interrupted -> $OUT"

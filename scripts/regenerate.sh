#!/usr/bin/env bash
# Full pipeline for every model in models.yaml:
#   fetch -> sweep -> context_length -> mlx_bench (if an MLX equivalent exists) -> summarize
#
#   scripts/regenerate.sh            # all models
#   scripts/regenerate.sh ID [ID..]  # subset
#
# Idempotent: every stage skips work whose output already exists, so an
# interrupted run can simply be restarted. Results land in
# results/<host-slug>/<model-id>/{sweep,context,mlx}/ and summarize.py then
# rewrites results/<host-slug>/summary.{md,json} and the README results block.
#
# Environment: SMOKE=1 shrinks every stage to a single tiny configuration and
# writes under results/smoke/ (gitignored) instead of results/<host>/.
#               SKIP_MLX=1 skips the MLX stage.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
# shellcheck source=lib/host.sh
source "$ROOT/scripts/lib/host.sh"

# Measurement conditions (see README, "Measurement conditions"): mains power,
# machine awake, nothing else running. Refuse on battery, and keep the machine
# from idle-sleeping by re-running this script under caffeinate.
require_ac_power
if [ -z "${ASSAY_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export ASSAY_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

# run_stage DIR CMD...: run a stage and record the conditions it ran under in
# DIR/stage-conditions.json. A record that already says the conditions were
# bad is never overwritten by a later clean run (delete the stage directory to
# redo it); an invocation that added no files leaves the record alone.
run_stage() {
  local dir="$1"; shift
  mkdir -p "$dir"
  if ! wait_for_quiet; then
    echo "machine not idle (CPU ${LAST_IDLE}% idle, need ${QUIET_IDLE_MIN:-85}%); not running $dir. Close other applications and run again." >&2
    return 3
  fi
  local sig_before sig_after pb sb pa sa slept=false rc=0 idle_before="$LAST_IDLE" quiet_met="$QUIET_MET"
  sig_before=$(find "$dir" -maxdepth 1 -type f ! -name stage-conditions.json -exec basename {} \; | sort | shasum | cut -d' ' -f1)
  pb=$(power_json); sb=$(sleep_stamp)
  "$@" || rc=$?
  pa=$(power_json); sa=$(sleep_stamp)
  [ "$sb" = "$sa" ] || slept=true
  sig_after=$(find "$dir" -maxdepth 1 -type f ! -name stage-conditions.json -exec basename {} \; | sort | shasum | cut -d' ' -f1)
  local rec="$dir/stage-conditions.json"
  if [ "$sig_before" != "$sig_after" ]; then
    local prev_dirty=false
    if [ -s "$rec" ] && [ "$(jq -r '(.power_before.source != "AC") or (.power_after.source != "AC") or .slept_during_run' "$rec")" = "true" ]; then
      prev_dirty=true
    fi
    if [ "$prev_dirty" = false ]; then
      jq -n --argjson before "$pb" --argjson after "$pa" --argjson slept "$slept" \
        --argjson idle "$idle_before" --argjson quiet_met "$quiet_met" --argjson quiet_min "${QUIET_IDLE_MIN:-85}" \
        '{power_before:$before, power_after:$after, slept_during_run:$slept, timed_out:false,
          cpu_idle_before_pct:$idle, quiet_threshold_pct:$quiet_min, quiet_threshold_met:$quiet_met}' > "$rec"
    fi
  fi
  return $rc
}

# Per-stage grid overrides. sweep.sh and context_length.sh read the same
# variable names (and PROMPTS has a different format in each), so one
# invocation cannot give them different grids through THREADS or PROMPTS.
# SWEEP_<VAR> and CONTEXT_<VAR> are passed to that stage only, as <VAR>.
# Unset means the stage's own default; every grid lands in its manifest.
stage_env() {
  local prefix="$1" v name
  STAGE_ENV=()
  for v in THREADS BATCHES FA KV PROMPTS GEN REPS; do
    name="${prefix}_${v}"
    if [ -n "${!name:-}" ]; then STAGE_ENV+=("$v=${!name}"); fi
  done
}

if [ "$#" -gt 0 ]; then IDS=("$@"); else
  IDS=()
  while IFS= read -r line; do IDS+=("$line"); done < <(python3 scripts/lib/modelsyaml.py ids)
fi

if [ "${SMOKE:-0}" = "1" ]; then HOST_DIR="results/smoke"; else HOST_DIR="results/$(host_slug)"; fi
mkdir -p "$HOST_DIR"
llama_cache_version
echo "host: $(host_chip), $(host_mem_gb) GB, macOS $(host_macos); llama.cpp $(llama_version_string)"
echo "results -> $HOST_DIR"

for id in "${IDS[@]}"; do
  file=$(python3 scripts/lib/modelsyaml.py get "$id" file)
  mlx=$(python3 scripts/lib/modelsyaml.py get "$id" mlx_equivalent)
  echo "=== $id"
  scripts/fetch.sh "$id"
  stage_env SWEEP
  env ${STAGE_ENV[@]+"${STAGE_ENV[@]}"} scripts/sweep.sh "models/$file" "$HOST_DIR/$id/sweep"
  stage_env CONTEXT
  run_stage "$HOST_DIR/$id/context" env ${STAGE_ENV[@]+"${STAGE_ENV[@]}"} scripts/context_length.sh "models/$file" "$HOST_DIR/$id/context"
  if [ -n "$mlx" ] && [ "${SKIP_MLX:-0}" != "1" ]; then
    if [ -x .venv/bin/python ]; then
      run_stage "$HOST_DIR/$id/mlx" .venv/bin/python scripts/mlx_bench.py --model "$mlx" --out "$HOST_DIR/$id/mlx" \
        ${SMOKE:+--smoke} || echo "mlx_bench failed for $id (continuing)"
    else
      echo "no .venv; skipping MLX. Create it with: python3 -m venv .venv && .venv/bin/pip install mlx-lm"
    fi
  fi
done

if [ "${SMOKE:-0}" = "1" ]; then
  python3 scripts/summarize.py "$HOST_DIR" --no-readme
else
  # Summarise every host under results/ so the README block keeps the other
  # hosts' tables when this one is regenerated.
  python3 scripts/summarize.py results
fi

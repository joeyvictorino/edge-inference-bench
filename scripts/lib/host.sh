#!/usr/bin/env bash
# Shared helpers: host fingerprint and manifest writer.
# Source this file; do not execute it.
#
#   host_slug            -> e.g. apple-a18-pro-8gb-macos27.0
#   host_fingerprint_json -> JSON object with chip/memory/macOS/llama.cpp fields
#   write_manifest OUT_DIR MODEL_PATH KIND  -> OUT_DIR/manifest.json
#   model_sha256 MODEL_PATH -> sha256 (from models.lock when it matches, else computed)

_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

host_chip() { sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown; }
host_mem_bytes() { sysctl -n hw.memsize 2>/dev/null || echo 0; }
host_mem_gb() { echo $(( $(host_mem_bytes) / 1024 / 1024 / 1024 )); }
host_macos() { sw_vers -productVersion 2>/dev/null || echo unknown; }
host_macos_build() { sw_vers -buildVersion 2>/dev/null || echo unknown; }
host_cores_total() { sysctl -n hw.ncpu 2>/dev/null || echo 0; }
host_cores_perf() { sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 0; }
host_cores_eff() { sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 0; }
host_model_id() { sysctl -n hw.model 2>/dev/null || echo unknown; }

llama_version_string() {
  # "version: 0.5.0 (build 11146, commit 7fe450e19)" style line from llama-bench
  llama-bench --version 2>&1 | grep -E '^version:' | head -1 | sed 's/^version: //' || echo unknown
}
llama_build_line() {
  llama-bench --version 2>&1 | grep -E '^built with' | head -1 || echo unknown
}
llama_metal_note() {
  # the tensor-API notice llama.cpp prints on this class of chip
  llama-bench --version 2>&1 | grep -E 'ggml_metal_device_init' | head -1 || true
}

host_slug() {
  local chip mem os
  chip=$(host_chip | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '-' | sed 's/-\+/-/g; s/^-//; s/-$//')
  mem=$(host_mem_gb)
  os=$(host_macos)
  echo "${chip}-${mem}gb-macos${os}"
}

model_sha256() {
  local path="$1" root base
  root=$(_repo_root)
  base=$(basename "$path")
  if [ -f "$root/models.lock" ]; then
    local locked
    locked=$(python3 - "$root/models.lock" "$base" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1]))
for entry in lock.values():
    if entry.get("file") == sys.argv[2]:
        print(entry.get("sha256", ""))
        break
PY
)
    if [ -n "$locked" ]; then echo "$locked"; return 0; fi
  fi
  shasum -a 256 "$path" | awk '{print $1}'
}

host_fingerprint_json() {
  jq -n \
    --arg chip "$(host_chip)" \
    --arg model_identifier "$(host_model_id)" \
    --argjson memory_bytes "$(host_mem_bytes)" \
    --argjson cores_total "$(host_cores_total)" \
    --argjson cores_performance "$(host_cores_perf)" \
    --argjson cores_efficiency "$(host_cores_eff)" \
    --arg macos "$(host_macos)" \
    --arg macos_build "$(host_macos_build)" \
    --arg llama_cpp_version "$(llama_version_string)" \
    --arg llama_cpp_built_with "$(llama_build_line)" \
    --arg metal_note "$(llama_metal_note)" \
    --arg host_slug "$(host_slug)" \
    '{chip:$chip, model_identifier:$model_identifier, memory_bytes:$memory_bytes,
      cores_total:$cores_total, cores_performance:$cores_performance, cores_efficiency:$cores_efficiency,
      macos:$macos, macos_build:$macos_build,
      llama_cpp_version:$llama_cpp_version, llama_cpp_built_with:$llama_cpp_built_with,
      metal_note:$metal_note, host_slug:$host_slug}'
}

# write_manifest OUT_DIR MODEL_PATH KIND [extra_json]
write_manifest() {
  local out="$1" model="$2" kind="$3" extra="${4:-{\}}"
  mkdir -p "$out"
  # A manifest describes the run that produced the files next to it. On a
  # resumed run keep the original (timestamp and all) so committed manifests
  # do not churn; delete it to force a rewrite.
  if [ -s "$out/manifest.json" ]; then
    echo "manifest.json exists in $out; keeping it"
    return 0
  fi
  host_fingerprint_json | jq \
    --arg kind "$kind" \
    --arg model_path "$model" \
    --arg model_file "$(basename "$model")" \
    --arg model_sha256 "$(model_sha256 "$model")" \
    --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson extra "$extra" \
    '{schema:"edge-inference-bench/manifest/v1", kind:$kind, engine:"llama.cpp",
      model_path:$model_path, model_file:$model_file, model_sha256:$model_sha256,
      timestamp:$timestamp, host:.} + $extra' > "$out/manifest.json"
}

# ---- measurement conditions -------------------------------------------------
# A throughput number is only meaningful if the machine was awake, on mains
# power and not otherwise occupied while it was measured. These helpers let the
# scripts refuse to run on battery and notice a sleep in the middle of a run.

power_source() {
  if pmset -g batt 2>/dev/null | head -1 | grep -q "AC Power"; then echo AC; else echo Battery; fi
}
battery_pct() { pmset -g batt 2>/dev/null | grep -Eo '[0-9]+%' | head -1 | tr -d '%'; }
# Changes whenever the machine goes to sleep; compare before and after a run.
# ASSAY_SLEEP_STAMP_CMD lets tests inject a command that simulates a sleep.
sleep_stamp() { ${ASSAY_SLEEP_STAMP_CMD:-sysctl -n kern.sleeptime} 2>/dev/null | tr -d '\n' || echo none; }

require_ac_power() {
  if [ "$(power_source)" != "AC" ] && [ "${ALLOW_BATTERY:-0}" != "1" ]; then
    echo "refusing to benchmark on battery ($(battery_pct)%): plug in, or set ALLOW_BATTERY=1" >&2
    echo "(runs made on battery are recorded as such and excluded from published summaries)" >&2
    exit 2
  fi
}

# power_json -> {"source":"AC|Battery","battery_pct":N}
power_json() {
  jq -n --arg source "$(power_source)" --argjson pct "$(battery_pct || echo null)" \
    '{source:$source, battery_pct:$pct}'
}

# ---- CPU contention ----------------------------------------------------------
# cpu_idle_pct: percentage of CPU idle over a 2-second window (top's second
# sample). ASSAY_CPU_IDLE_CMD lets tests inject a number.
cpu_idle_pct() {
  if [ -n "${ASSAY_CPU_IDLE_CMD:-}" ]; then ${ASSAY_CPU_IDLE_CMD}; return; fi
  top -l 2 -n 0 -s 2 2>/dev/null | grep 'CPU usage' | tail -1 | sed -E 's/.* ([0-9.]+)% idle.*/\1/'
}

# wait_for_quiet: poll until the machine is idle enough to benchmark. Sets
# LAST_IDLE to the last sample. Returns 1 if it is still busy after
# QUIET_TIMEOUT seconds (default 900). QUIET_IDLE_MIN (default 85) is the
# required idle percentage; QUIET_POLL (default 15) the gap between samples.
LAST_IDLE=0
wait_for_quiet() {
  local min="${QUIET_IDLE_MIN:-85}" limit="${QUIET_TIMEOUT:-900}" poll="${QUIET_POLL:-15}" waited=0 idle
  while :; do
    idle=$(cpu_idle_pct); idle=${idle:-0}
    LAST_IDLE=$idle
    if awk -v i="$idle" -v m="$min" 'BEGIN{exit !(i>=m)}'; then return 0; fi
    [ "$waited" -ge "$limit" ] && return 1
    sleep "$poll"; waited=$((waited + poll))
  done
}

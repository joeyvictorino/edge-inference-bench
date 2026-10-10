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

# _sysctl_int NAME -> the integer value, or 0 when the key is missing or empty
# (a GitHub runner VM has no hw.perflevel* keys; the value feeds jq --argjson).
_sysctl_int() {
  local v
  v=$(sysctl -n "$1" 2>/dev/null) || v=""
  case "$v" in ''|*[!0-9]*) echo 0 ;; *) echo "$v" ;; esac
}
host_chip() { sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown; }
host_mem_bytes() { _sysctl_int hw.memsize; }
host_mem_gb() { echo $(( $(host_mem_bytes) / 1024 / 1024 / 1024 )); }
host_macos() { sw_vers -productVersion 2>/dev/null || echo unknown; }
host_macos_build() { sw_vers -buildVersion 2>/dev/null || echo unknown; }
host_cores_total() { _sysctl_int hw.ncpu; }
host_cores_perf() { _sysctl_int hw.perflevel0.physicalcpu; }
host_cores_eff() { _sysctl_int hw.perflevel1.physicalcpu; }
host_model_id() { sysctl -n hw.model 2>/dev/null || echo unknown; }

# The version output is captured once per process (call llama_cache_version
# before using the readers below from subshells): each llama.cpp start compiles
# the Metal library, which takes about 40 s on a GitHub runner. Older
# llama-bench builds (Homebrew 0.4.0) reject --version; llama-cli from the
# same installation reports the same build, so it is asked as a fallback.
llama_cache_version() {
  if [ -z "${_LLAMA_VERSION_OUT+x}" ]; then
    _LLAMA_VERSION_OUT=$(llama-bench --version 2>&1 || true)
    if ! grep -qE '^version:' <<<"$_LLAMA_VERSION_OUT" && command -v llama-cli >/dev/null 2>&1; then
      _LLAMA_VERSION_OUT="$_LLAMA_VERSION_OUT
$(llama-cli --version 2>&1 || true)"
    fi
  fi
}
_llama_version_out() { llama_cache_version; printf '%s\n' "$_LLAMA_VERSION_OUT"; }
llama_version_string() {
  # "version: 0.5.0 (build 11146, commit 7fe450e19)" style line
  _llama_version_out | grep -E '^version:' | head -1 | sed 's/^version: //' || echo unknown
}
llama_build_line() {
  _llama_version_out | grep -E '^built with' | head -1 || echo unknown
}
llama_metal_note() {
  # the tensor-API notice llama.cpp prints on this class of chip
  _llama_version_out | grep -E 'ggml_metal_device_init' | head -1 || true
}
# Homebrew formula versions, when llama.cpp came from Homebrew ("" otherwise).
llama_brew_versions() {
  command -v brew >/dev/null 2>&1 || return 0
  brew list --versions llama.cpp ggml 2>/dev/null | tr '\n' ';' | sed 's/;$//' || true
}

# ---- host kind ---------------------------------------------------------------
# BENCH_HOST says what kind of machine this is. Unset (the default) means the
# operator's own Mac, and every measurement-condition check below applies
# unchanged. BENCH_HOST=gha-macos marks a GitHub-hosted macOS runner: a virtual
# machine on a data-centre Mac, shared hardware, no battery. Results from it go
# under a host directory starting with "gha-runner-" and their manifests carry
# the runner image and the Actions run URL, so they can never be mistaken for
# the operator's machine. Any other value is refused.
host_kind() {
  case "${BENCH_HOST:-}" in
    "") echo local ;;
    gha-macos) echo gha-macos ;;
    *) echo "unknown BENCH_HOST '${BENCH_HOST}' (expected unset or gha-macos)" >&2; return 2 ;;
  esac
}
is_gha_host() { [ "$(host_kind)" = "gha-macos" ]; }

host_slug() {
  local chip mem os prefix=""
  host_kind >/dev/null || return 2
  chip=$(host_chip | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '-' | tr -s '-' | sed 's/^-//; s/-$//')
  mem=$(host_mem_gb)
  os=$(host_macos)
  if is_gha_host; then prefix="gha-runner-"; fi
  echo "${prefix}${chip}-${mem}gb-macos${os}"
}

# runner_json -> GitHub runner facts in gha-macos mode, null otherwise.
runner_json() {
  if ! is_gha_host; then echo null; return 0; fi
  local url=""
  if [ -n "${GITHUB_RUN_ID:-}" ]; then
    url="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-}/actions/runs/${GITHUB_RUN_ID}/attempts/${GITHUB_RUN_ATTEMPT:-1}"
  fi
  jq -n --arg image_os "${ImageOS:-unknown}" --arg image_version "${ImageVersion:-unknown}" \
    --arg arch "${RUNNER_ARCH:-unknown}" --arg label "${BENCH_RUNNER_LABEL:-unknown}" --arg run_url "$url" \
    '{virtualized:true, image_os:$image_os, image_version:$image_version, runner_arch:$arch,
      runs_on:$label, run_url:$run_url}'
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
  llama_cache_version
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
    --arg llama_cpp_homebrew "$(llama_brew_versions)" \
    --arg host_slug "$(host_slug)" \
    --arg host_kind "$(host_kind)" \
    --argjson runner "$(runner_json)" \
    '{chip:$chip, model_identifier:$model_identifier, memory_bytes:$memory_bytes,
      cores_total:$cores_total, cores_performance:$cores_performance, cores_efficiency:$cores_efficiency,
      macos:$macos, macos_build:$macos_build,
      llama_cpp_version:$llama_cpp_version, llama_cpp_built_with:$llama_cpp_built_with,
      metal_note:$metal_note, llama_cpp_homebrew:$llama_cpp_homebrew, host_slug:$host_slug, host_kind:$host_kind, runner:$runner}'
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

# Output is captured before it is inspected: piping pmset into head and grep
# under `set -o pipefail` can report a spurious failure when pmset is still
# writing after head has closed the pipe, which would misreport mains power as
# battery.
power_source() {
  local out first
  out=$(pmset -g batt 2>/dev/null) || true
  first=${out%%$'\n'*}
  case "$first" in *"AC Power"*) echo AC ;; *) echo Battery ;; esac
}
battery_pct() {
  local out
  out=$(pmset -g batt 2>/dev/null) || true
  awk 'match($0, /[0-9]+%/) { print substr($0, RSTART, RLENGTH - 1); exit }' <<<"$out"
}
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
  local pct
  pct=$(battery_pct || true)
  [ -n "$pct" ] || pct=null   # no battery line (desktop Mac or virtual machine)
  jq -n --arg source "$(power_source)" --argjson pct "$pct" \
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
# LAST_IDLE to the last sample and QUIET_MET to true or false. Returns 1 if it
# is still busy after QUIET_TIMEOUT seconds (default 900). QUIET_IDLE_MIN
# (default 85) is the required idle percentage; QUIET_POLL (default 15) the gap
# between samples. LAST_IDLE and QUIET_MET are read by the scripts that source
# this file, so they are exported.
#
# On a GitHub runner (BENCH_HOST=gha-macos) the 3-vCPU virtual machine never
# came close to 85% idle in the probe runs (0-73% idle; docs/notes.md). There the
# wait is shortened (QUIET_TIMEOUT default 120) and, instead of refusing, the
# function returns 0 with QUIET_MET=false; the scripts record that in every
# conditions file and summarize.py reports how many runs started below the
# threshold. On the operator's own machine nothing changes: a busy machine is
# refused.
export LAST_IDLE=0
export QUIET_MET=false
wait_for_quiet() {
  local min="${QUIET_IDLE_MIN:-85}" limit="${QUIET_TIMEOUT:-900}" poll="${QUIET_POLL:-15}" waited=0 idle gha=false
  if is_gha_host; then gha=true; limit="${QUIET_TIMEOUT:-120}"; fi
  while :; do
    idle=$(cpu_idle_pct); idle=${idle:-0}
    export LAST_IDLE="$idle"
    if awk -v i="$idle" -v m="$min" 'BEGIN{exit !(i>=m)}'; then export QUIET_MET=true; return 0; fi
    if [ "$waited" -ge "$limit" ]; then
      export QUIET_MET=false
      if [ "$gha" = true ]; then
        echo "  runner never reached ${min}% idle (last ${idle}%); recording that and running (BENCH_HOST=gha-macos)"
        return 0
      fi
      return 1
    fi
    sleep "$poll"; waited=$((waited + poll))
  done
}

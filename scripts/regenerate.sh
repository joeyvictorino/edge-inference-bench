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

if [ "$#" -gt 0 ]; then IDS=("$@"); else
  IDS=()
  while IFS= read -r line; do IDS+=("$line"); done < <(python3 scripts/lib/modelsyaml.py ids)
fi

if [ "${SMOKE:-0}" = "1" ]; then HOST_DIR="results/smoke"; else HOST_DIR="results/$(host_slug)"; fi
mkdir -p "$HOST_DIR"
echo "host: $(host_chip), $(host_mem_gb) GB, macOS $(host_macos); llama.cpp $(llama_version_string)"
echo "results -> $HOST_DIR"

for id in "${IDS[@]}"; do
  file=$(python3 scripts/lib/modelsyaml.py get "$id" file)
  mlx=$(python3 scripts/lib/modelsyaml.py get "$id" mlx_equivalent)
  echo "=== $id"
  scripts/fetch.sh "$id"
  scripts/sweep.sh "models/$file" "$HOST_DIR/$id/sweep"
  scripts/context_length.sh "models/$file" "$HOST_DIR/$id/context"
  if [ -n "$mlx" ] && [ "${SKIP_MLX:-0}" != "1" ]; then
    if [ -x .venv/bin/python ]; then
      .venv/bin/python scripts/mlx_bench.py --model "$mlx" --out "$HOST_DIR/$id/mlx" \
        ${SMOKE:+--smoke} || echo "mlx_bench failed for $id (continuing)"
    else
      echo "no .venv; skipping MLX. Create it with: python3 -m venv .venv && .venv/bin/pip install mlx-lm"
    fi
  fi
done

if [ "${SMOKE:-0}" = "1" ]; then
  python3 scripts/summarize.py "$HOST_DIR" --no-readme
else
  python3 scripts/summarize.py "$HOST_DIR"
fi

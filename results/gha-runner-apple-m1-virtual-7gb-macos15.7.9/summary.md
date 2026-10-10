# Results: gha-runner-apple-m1-virtual-7gb-macos15.7.9

> **GitHub-hosted runner, not a physical machine.** These numbers come from a virtual machine (`runs-on: macos-15`, image macos15 20260907.0337.1) on shared data-centre hardware. They describe that runner class, not any laptop, and are not comparable with the other hosts in this repository.

Host: Apple M1 (Virtual), 7 GB memory, 3 cores (3 performance + 0 efficiency), macOS 15.7.9 (24G830). llama.cpp 0.4.0 (build 10809, commit 5266f24da).
Produced by GitHub Actions run https://github.com/joeyvictorino/edge-inference-bench/actions/runs/38036322821/attempts/1.
Metal note recorded by llama.cpp: `ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices`.
Data collected between 2026-10-10T08:00:33Z and 2026-10-10T10:59:28Z (UTC).

> **Not reproduced (`qwen2.5-1.5b-instruct-q4_k_m`).** A second run on another runner (`run-38036322821-replica2`) matched 13 of 64 metrics within the published inter-quartile range; median difference 18.5%, largest 53.0%. Treat these numbers as one observation, not a benchmark result.

> **Not reproduced (`qwen2.5-3b-instruct-q4_k_m`).** A second run on another runner (`run-38036322821-replica2`) matched 5 of 64 metrics within the published inter-quartile range; median difference 28.5%, largest 183.3%. Treat these numbers as one observation, not a benchmark result.

Values are the median of the per-repetition tokens/s samples with the inter-quartile range in parentheses; `(n=1)` marks a single repetition; `n/a` means no completed run.

## Best llama.cpp configuration per model

Best = highest median generation tokens/s across the sweep grid. Prompt columns are the same configuration's prompt-processing throughput.

| Model | Configuration | Generation t/s | Prompt 512 t/s | Prompt 2048 t/s | Configs ok/failed/excluded |
|---|---|---|---|---|---|
| `qwen2.5-1.5b-instruct-q4_k_m` | t=2 b=512 ub=256 fa=off kv=f16 | 7.6 (IQR 0.1) (tg128) | 62.6 (IQR 0.9) | 33.5 (IQR 0.0) | 18/6/0 |
| `qwen2.5-3b-instruct-q4_k_m` | t=2 b=1024 ub=512 fa=on kv=q8_0 | 5.4 (IQR 0.3) (tg128) | 31.0 (IQR 1.1) | 27.6 (IQR 1.9) | 18/6/0 |

Started below the CPU-idle threshold (recorded, not refused, on a GitHub runner; the `cpu_idle_before_pct` in each conditions file gives the level): `qwen2.5-3b-instruct-q4_k_m`: 2 of 18 sweep configurations.

## Throughput versus prompt length (llama.cpp)

Prompt-processing tokens/s by prompt length (llama-bench defaults apart from `-p`; see `context/manifest.json` for the exact grid).

| Prompt tokens | `qwen2.5-1.5b-instruct-q4_k_m` | `qwen2.5-3b-instruct-q4_k_m` |
|---|---|---|
| 512 | 62.1 (IQR 1.3) | 24.5 (IQR 1.3) |
| 1024 | 53.5 (IQR 12.1) | 28.8 (IQR 2.3) |
| 2048 | 56.0 (IQR 1.9) | 23.7 (IQR 3.5) |
| 4096 | 33.5 (IQR 0.6) | 17.3 (IQR 0.5) |
| 8192 | 21.7 (IQR 0.9) | 9.3 (IQR 0.9) |

Generation tokens/s (64 tokens) measured in the same runs:

| Prompt tokens | `qwen2.5-1.5b-instruct-q4_k_m` | `qwen2.5-3b-instruct-q4_k_m` |
|---|---|---|
| 512 | 4.3 (IQR 0.4) | 2.5 (IQR 0.2) |
| 1024 | 5.1 (IQR 0.6) | 4.4 (IQR 0.2) |
| 2048 | 6.7 (IQR 0.2) | 1.9 (IQR 0.3) |
| 4096 | 7.1 (IQR 1.5) | 2.7 (IQR 0.4) |
| 8192 | 5.0 (IQR 0.8) | 2.1 (IQR 0.1) |

## Reproducibility

Each replica is a second run of the same pipeline, compared with `scripts/compare_runs.py`. Rule: a metric reproduces when the replica's median is within the published run's inter-quartile range. Full tables: `results/replicas/gha-runner-apple-m1-virtual-7gb-macos15.7.9/<replica>/compare.md`.

| Replica | Model | Metrics compared | Reproduced | Median abs. difference | Largest abs. difference |
|---|---|---|---|---|---|
| `run-38036322821-replica2` | `qwen2.5-1.5b-instruct-q4_k_m` | 64 | 13 | 18.5% | 53.0% |
| `run-38036322821-replica2` | `qwen2.5-3b-instruct-q4_k_m` | 64 | 5 | 28.5% | 183.3% |

## llama.cpp versus MLX, same base model

_No MLX runs found._

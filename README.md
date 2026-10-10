# edge-inference-bench

Reproducible latency and throughput numbers for small open-weight language
models (1.5B to 4B parameters, 4-bit quantised), measured with llama.cpp and
MLX, each set tied to the one host that produced it. Two hosts are defined:
the author's low-memory laptop-class machine (results pending; see
"Measurement conditions") and a GitHub-hosted Apple-silicon runner, a virtual
machine anyone can rerun from this repository (see "GitHub runner results").

The intended use is as **priors for an agent harness**: when a tool-using
agent loop runs a small local model, how long does a 2k-token prompt take to
process, how many tokens per second come back, and how does that change as
context grows? Each results table answers that for its own host only.

## What is measured

| Script | What it does | Repetitions |
|---|---|---|
| `scripts/sweep.sh` | `llama-bench` over a grid of threads {2, 4, 6}, batch/micro-batch {256, 512, 1024} x {128, 256, 512} (micro-batch <= batch), flash attention {off, on}, KV cache type {f16, q8_0}; prompt sizes 512, 2048, 8192 tokens and 128 generated tokens per configuration | 3 |
| `scripts/context_length.sh` | prompt-processing and generation throughput versus prompt length: 512, 1024, 2048, 4096, 8192, 16384 tokens with 64 generated tokens, llama-bench defaults otherwise | 3 |
| `scripts/mlx_bench.py` | the same prompt lengths through `mlx_lm` with the 4-bit MLX conversion of the same base model, 128 generated tokens, recording prompt tokens/s, generation tokens/s, time-to-first-token and peak Metal memory | 3 |

Every reported value is the **median** of the per-repetition tokens/s samples
with the **inter-quartile range** alongside. Means are not reported. Each
configuration's raw `llama-bench -o json` output (or `mlx_bench.py` JSON) is
committed under `results/<host>/<model>/`, together with a `manifest.json`
holding the host fingerprint (chip, memory, macOS, llama.cpp version string,
timestamp, model sha256). `scripts/summarize.py` regenerates every table from
those files; nothing in the results section is typed by hand.

Configurations that cannot run (for example a quantised V cache with flash
attention off, or a prompt that does not fit in memory) are recorded as
`*.failed.json` and counted, never silently dropped.

### What is not claimed

- **No cross-machine comparison.** Each host has its own table, llama.cpp
  build and MLX version. The numbers are not a ranking of chips, and the
  llama.cpp-vs-MLX table is a same-machine, same-base-model comparison only.
- **The GitHub runner is not physical Apple silicon.** It is a virtual machine
  with a paravirtual Metal GPU. Its numbers describe that runner class and
  must not be read as the performance of an M1 or of any laptop.
- **No multi-token prediction (MTP) or speculative decoding results.**
  Everything is plain autoregressive decoding.
- **No fine-tuning, no quality evaluation.** Only throughput and latency are
  measured; output quality of the quantised models is out of scope.
- **No server-level numbers.** `llama-bench` measures the engine; HTTP
  overhead, batching across requests and KV-cache reuse between turns are not
  included.

## Hardware and software

The author's machine, recorded from the system on 2026-10-07 (see
`docs/notes.md` for the exact commands). No results from it are published
yet:

- Chip: Apple A18 Pro (`machdep.cpu.brand_string`), model identifier `Mac17,5`
- Cores: 6 (2 performance, 4 efficiency)
- Memory: 8 GB unified
- OS: macOS 27.0 (build 26A428)
- llama.cpp: `version: 0.5.0 (build 11146, commit 7fe450e19)`, built with
  AppleClang 21.0.0.21000334 for Darwin arm64 (Homebrew `llama.cpp`, ggml 0.25.1)
- Python 3.14.4; mlx 0.32.3, mlx-lm 0.32.0 in `.venv`

Every manifest re-records these values at run time, so a result file always
carries the versions it was produced with.

## Models

Listed in `models.yaml` (five 4-bit GGUFs with their MLX 4-bit equivalents).
`scripts/fetch.sh` downloads one model into `models/` (gitignored) and writes
its real sha256 and byte size to `models.lock`; the manifests reference that
digest.

## How to regenerate

Requirements: macOS with Homebrew `llama.cpp` (`llama-bench` on `PATH`),
`jq`, `curl`, Python 3.10+, and for the MLX stage a local venv:

    python3 -m venv .venv && .venv/bin/pip install mlx-lm

Then:

    scripts/regenerate.sh                 # all models: fetch -> sweep -> context -> mlx -> summarize
    scripts/regenerate.sh <model-id>      # one model (ids: python3 scripts/lib/modelsyaml.py ids)
    python3 scripts/summarize.py results  # tables only, from committed JSON

Every stage is resumable and idempotent: existing result files are skipped,
so an interrupted sweep can simply be restarted. `SMOKE=1 scripts/regenerate.sh`
runs one tiny configuration per stage into `results/smoke/` (gitignored) to
check the pipeline without producing publishable numbers.

Run the sweep on an otherwise idle machine; llama-bench measures wall-clock
throughput and competes with anything else using the GPU or memory bandwidth.

## Results

<!-- RESULTS:BEGIN -->
Generated by `scripts/summarize.py`; do not edit by hand. Full tables: [`results/gha-runner-apple-m1-virtual-7gb-macos15.7.9/summary.md`](results/gha-runner-apple-m1-virtual-7gb-macos15.7.9/summary.md).

> **GitHub-hosted runner, not a physical machine.** These numbers come from a virtual machine (`runs-on: macos-15`, image macos15 20260907.0337.1) on shared data-centre hardware. They describe that runner class, not any laptop, and are not comparable with the other hosts in this repository.

Host: Apple M1 (Virtual), 7 GB memory, 3 cores (3 performance + 0 efficiency), macOS 15.7.9 (24G830). llama.cpp 0.4.0 (build 10809, commit 5266f24da).
Produced by GitHub Actions run https://github.com/joeyvictorino/edge-inference-bench/actions/runs/38036322821/attempts/1.
Metal note recorded by llama.cpp: `ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices`.
Data collected between 2026-10-10T08:00:33Z and 2026-10-10T10:59:28Z (UTC).

> **Not reproduced (`qwen2.5-1.5b-instruct-q4_k_m`).** A second run on another runner (`run-38036322821-replica2`) matched 13 of 64 metrics within the published inter-quartile range; median difference 18.5%, largest 53.0%. Treat these numbers as one observation, not a benchmark result.

> **Not reproduced (`qwen2.5-3b-instruct-q4_k_m`).** A second run on another runner (`run-38036322821-replica2`) matched 5 of 64 metrics within the published inter-quartile range; median difference 28.5%, largest 183.3%. Treat these numbers as one observation, not a benchmark result.

Values are the median of the per-repetition tokens/s samples with the inter-quartile range in parentheses; `(n=1)` marks a single repetition; `n/a` means no completed run.

### Best llama.cpp configuration per model

Best = highest median generation tokens/s across the sweep grid. Prompt columns are the same configuration's prompt-processing throughput.

| Model | Configuration | Generation t/s | Prompt 512 t/s | Prompt 2048 t/s | Configs ok/failed/excluded |
|---|---|---|---|---|---|
| `qwen2.5-1.5b-instruct-q4_k_m` | t=2 b=512 ub=256 fa=off kv=f16 | 7.6 (IQR 0.1) (tg128) | 62.6 (IQR 0.9) | 33.5 (IQR 0.0) | 18/6/0 |
| `qwen2.5-3b-instruct-q4_k_m` | t=2 b=1024 ub=512 fa=on kv=q8_0 | 5.4 (IQR 0.3) (tg128) | 31.0 (IQR 1.1) | 27.6 (IQR 1.9) | 18/6/0 |

Started below the CPU-idle threshold (recorded, not refused, on a GitHub runner; the `cpu_idle_before_pct` in each conditions file gives the level): `qwen2.5-3b-instruct-q4_k_m`: 2 of 18 sweep configurations.

### Throughput versus prompt length (llama.cpp)

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

### Reproducibility

Each replica is a second run of the same pipeline, compared with `scripts/compare_runs.py`. Rule: a metric reproduces when the replica's median is within the published run's inter-quartile range. Full tables: `results/replicas/gha-runner-apple-m1-virtual-7gb-macos15.7.9/<replica>/compare.md`.

| Replica | Model | Metrics compared | Reproduced | Median abs. difference | Largest abs. difference |
|---|---|---|---|---|---|
| `run-38036322821-replica2` | `qwen2.5-1.5b-instruct-q4_k_m` | 64 | 13 | 18.5% | 53.0% |
| `run-38036322821-replica2` | `qwen2.5-3b-instruct-q4_k_m` | 64 | 5 | 28.5% | 183.3% |

### llama.cpp versus MLX, same base model

_No MLX runs found._
<!-- RESULTS:END -->

## GitHub runner results

Because the author's machine has not yet had a quiet, mains-powered window
for the full sweep, the same scripts also run on a GitHub-hosted macOS runner
through `.github/workflows/bench.yml`. Anyone with write access to a fork can
rerun it (`gh workflow run bench.yml`), and every manifest records the
runner image and the Actions run URL that produced it.

What that runner is, as recorded by the probe and by every manifest
(`runs-on: macos-15`): `sysctl` reports `Apple M1 (Virtual)`, model
identifier `VirtualMac2,1`, 3 logical CPUs and 7 GiB of memory; macOS 15.7;
llama.cpp from Homebrew; Metal reports `MTL0 (Apple Paravirtual device)`. It
is a virtual machine on shared data-centre hardware.

How the method is adapted, explicitly (`BENCH_HOST=gha-macos`, see
`scripts/lib/host.sh`):

- Results go under `results/gha-runner-<chip>-<mem>gb-macos<ver>/`, so they
  can never be filed under the author's machine.
- The power and sleep checks run unchanged. The guest OS reports
  `AC Power`; that says nothing about the host's power, only that there is no
  battery to throttle.
- The CPU-idle check still runs, but it cannot refuse. Idle samples taken on
  the probe runners right after setup ranged from 0% to 73%
  (`docs/notes.md`), and a runner gives no way to wait for other tenants.
  In this mode the scripts wait up to 30 seconds for 85% idle and, if it is
  not reached, run anyway and record `cpu_idle_before_pct` and
  `quiet_threshold_met: false` in that conditions file. `summarize.py`
  states how many published runs started below the threshold. On the
  author's machine a busy CPU is still refused.
- The idle check only sees the virtual machine's own CPU. It cannot see
  other work on the same physical machine or its GPU.
- The grid is reduced, because one full-grid configuration (prompts of 512,
  2048 and 8192 tokens, 128 generated tokens, three repetitions) took 22 to 25
  minutes on each of three probe runners, and the full grid has 96. Sweep: threads {2, 3}; batch/micro-batch 512:256, 512:512, 1024:512;
  flash attention {off, on}; KV cache {f16, q8_0}; prompts of 512 and 2048
  tokens; 128 generated tokens; 3 repetitions. Context length: 512 to 8192
  tokens (no 16384), 64 generated tokens, 3 repetitions. MLX is not run. The
  exact grid is in each `manifest.json`.
- Each model runs twice, on two separate runners in the same workflow run.
  The first is published; the second is kept under `results/replicas/` and
  `scripts/compare_runs.py` checks it against the first with this
  repository's reproducibility rule: a metric reproduces when the replica's
  median is within the published run's inter-quartile range. The comparison
  is committed beside the replica.

Outcome of the first complete run (Actions run 38036322821, 2026-10-10): the
replicas did **not** reproduce the published medians under that rule, for
either model. The Results section below carries the counts, generated from
the committed comparison. The two replicas ran the same image, llama.cpp
build, model files and grid, so the spread is between runner instances, not
between configurations; it stays when only configurations that started at
85% idle or more in both runs are compared (`docs/notes.md`). These tables
are therefore published as observations of one shared virtual machine. They
are not a benchmark of Apple silicon, and they are not the result this
repository was built for, which is still the author's-machine sweep.

## Measurement conditions

A throughput number only means something if the machine was awake, on mains
power and not doing other work while it was measured. The scripts enforce the
first two and record them, and the third is on the operator.

- `regenerate.sh` and `sweep.sh` refuse to start on battery. `ALLOW_BATTERY=1`
  overrides that, but every result is then marked as made on battery.
- `regenerate.sh` runs under `caffeinate` so the machine does not idle-sleep.
- Each sweep configuration writes `<name>.env.json` (power source and battery
  level before and after, wall time, whether it timed out, whether the machine
  slept during it). The context-length and MLX stages write one
  `stage-conditions.json` each.
- A configuration during which the machine slept is kept as
  `<name>.interrupted.json`, never counted, and retried on the next run.
- A configuration still running after `CONFIG_TIMEOUT` seconds (default 1800)
  is killed and recorded as failed.
- `summarize.py` publishes a configuration or stage only if it ran on mains
  power, did not sleep, did not time out, and has a conditions record. It
  reports how many were excluded. A stage record that says the conditions were
  bad is never overwritten by a later clean run; delete the stage directory to
  redo it.
- Before each configuration (and each context-length or MLX stage) the scripts
  wait up to `QUIET_TIMEOUT` seconds (default 900) for the CPU to be at least
  `QUIET_IDLE_MIN` percent idle (default 85) and record the level as
  `cpu_idle_before_pct`. If the machine never quiets down, that configuration
  is not run and the script exits 3. Close other applications first. The check
  happens at the start of a configuration, so work that starts during one is
  not detected; the repetition spread (IQR) shown with every number is the
  second check, and a wide spread is a reason to rerun that configuration.

The first sweep on this repository's author's machine was run on battery with
repeated sleeps and other work in the background. Its results were discarded
rather than published; see `docs/notes.md`.

## Limitations

- **8 GB unified memory.** The GPU working-set ceiling reported by MLX is
  about 5.3 GiB. The 3B/4B models at 4-bit fit, but long prompts (8k to 16k
  tokens) may not; such points are recorded as failures rather than
  extrapolated.
- **Tensor API disabled.** llama.cpp prints
  `ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices`
  on this chip. The Metal backend runs without it, and the string is stored
  in every manifest. Newer chips with the tensor API enabled will behave
  differently; do not generalise from these numbers.
- **Thread count on a 2+4 core part.** `-t 6` schedules onto efficiency
  cores; with full GPU offload the thread setting mostly affects CPU-side
  work, so differences between thread counts are expected to be small.
- **Synthetic prompts.** `llama-bench` and `mlx_bench.py` time prompts of
  exactly N tokens, not real conversations; prompt content does not affect
  throughput materially, but prompt-cache reuse between agent turns is not
  modelled.
- **Thermal state is not controlled.** Repetitions are back to back; the
  IQR gives some indication of drift, but a passively cooled machine may
  throttle during a long sweep.

## Repository layout

    models.yaml              candidate models (repo, file, MLX equivalent)
    models.lock              sha256 and size of every downloaded GGUF (written by fetch.sh)
    scripts/fetch.sh         download one GGUF, update models.lock
    scripts/sweep.sh         llama-bench configuration grid -> raw JSON per config
    scripts/context_length.sh throughput vs prompt length -> raw JSON per point
    scripts/mlx_bench.py     same prompt lengths through mlx_lm
    scripts/summarize.py     median/IQR tables, summary.{md,json}, README block
    scripts/compare_runs.py  reproducibility check of a replica run against the published one
    scripts/regenerate.sh    whole pipeline for every model
    scripts/name-audit.sh    CI guard against names that must not appear here
    scripts/lib/             host fingerprint and models.yaml helpers
    results/<host>/<model>/  committed raw JSON (sweep/, context/, mlx/) and summaries
    results/replicas/        second runs kept for the reproducibility check (not summarised)
    .github/workflows/bench.yml  GitHub macOS runner benchmark (BENCH_HOST=gha-macos)
    tests/                   unit tests for summarize.py with a synthetic fixture
    docs/notes.md            environment observations and method notes

## License

Apache License 2.0; see `LICENSE`.

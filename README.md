# edge-inference-bench

Reproducible latency and throughput numbers for small open-weight language
models (1.5B to 4B parameters, 4-bit quantised) on one specific low-memory
laptop-class machine, measured with llama.cpp and MLX.

The intended use is as **priors for an agent harness**: when a tool-using
agent loop runs a small local model, how long does a 2k-token prompt take to
process, how many tokens per second come back, and how does that change as
context grows? These numbers answer that for this machine only.

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

- **No cross-machine comparison.** One host, one llama.cpp build, one MLX
  version. The numbers are not a ranking of chips, and the llama.cpp-vs-MLX
  table is a same-machine, same-base-model comparison only.
- **No multi-token prediction (MTP) or speculative decoding results.**
  Everything is plain autoregressive decoding.
- **No fine-tuning, no quality evaluation.** Only throughput and latency are
  measured; output quality of the quantised models is out of scope.
- **No server-level numbers.** `llama-bench` measures the engine; HTTP
  overhead, batching across requests and KV-cache reuse between turns are not
  included.

## Hardware and software

Recorded from the system on 2026-10-07 (see `docs/notes.md` for the exact
commands):

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
_Results pending._ No sweep has been run on this machine yet. This block is rewritten by `scripts/summarize.py` from the raw JSON under `results/`; it is never edited by hand.
<!-- RESULTS:END -->

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
    scripts/regenerate.sh    whole pipeline for every model
    scripts/name-audit.sh    CI guard against names that must not appear here
    scripts/lib/             host fingerprint and models.yaml helpers
    results/<host>/<model>/  committed raw JSON (sweep/, context/, mlx/) and summaries
    tests/                   unit tests for summarize.py with a synthetic fixture
    docs/notes.md            environment observations and method notes

## License

Apache License 2.0; see `LICENSE`.

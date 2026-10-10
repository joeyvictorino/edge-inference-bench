# Method notes and environment observations

Observations recorded while building and smoke-testing the pipeline on the
target machine. Nothing here is a benchmark result.

## Hardware and software captured from the system (2026-10-07)

| Item | Command | Value recorded |
|---|---|---|
| Chip | `sysctl -n machdep.cpu.brand_string` | `Apple A18 Pro` |
| Model identifier | `system_profiler SPHardwareDataType` | `Mac17,5` |
| Cores | `system_profiler SPHardwareDataType` | 6 (2 performance, 4 efficiency) |
| Memory | `system_profiler SPHardwareDataType` / `sysctl -n hw.memsize` | 8 GB (8589934592 bytes) |
| OS | `sw_vers` | macOS 27.0, build 26A428 |
| llama.cpp | `llama-server --version` (Homebrew) | `version: 0.5.0 (build 11146, commit 7fe450e19)`, `built with AppleClang 21.0.0.21000334 for Darwin arm64` |
| ggml | Homebrew keg path printed by `llama-bench` | `ggml 0.25.1` (`libggml-blas.so` loaded as BLAS backend; Metal backend `MTL`) |
| Python | `python3 --version` | Python 3.14.4 |
| mlx / mlx-lm | `.venv/bin/pip list` | mlx 0.32.3, mlx-metal 0.32.3, mlx-lm 0.32.0 (transformers 5.19.0, numpy 2.5.3) |
| Shell | `bash --version` | GNU bash 3.2.57 (the macOS system bash; scripts avoid bash 4 features) |

## Metal tensor API notice

Every `llama-bench` invocation on this chip prints:

    ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices

llama.cpp's Metal backend therefore runs without the tensor API on this
hardware. The string is captured verbatim into every `manifest.json`
(`host.metal_note`) so that the numbers stay tied to that fact. Numbers from
this machine should not be read as representative of newer chips.

MLX reports the device as `applegpu_g17p` with
`max_recommended_working_set_size` = 5726633984 bytes (about 5.3 GiB) out of
8 GiB of unified memory (`mx.device_info()`). That working-set ceiling, not
the nominal 8 GB, bounds what fits on the GPU side.

## llama-bench flags in the installed build

Checked with `llama-bench --help` on the installed version:

- `-fa` accepts `on|off|auto` (not `0|1`). The scripts pass `on`/`off` and
  encode it as `fa1`/`fa0` in the result file names; the raw JSON carries
  `"flash_attn": 1/0` as llama-bench writes it.
- `-ctk`/`-ctv` set K and V cache types; the sweep sets both to the same
  value (`f16` or `q8_0`).
- `-t`, `-b`, `-ub`, `-p`, `-n`, `-r`, `-o json` behave as expected. With
  `-p 512,2048,8192 -n 128` one invocation emits four test records (three
  prompt-processing, one generation); the sweep keeps one such JSON array per
  configuration.
- `-r 3` yields three entries in `samples_ts`, which is what `summarize.py`
  takes the median and IQR over. Default `-ngl -1` offloads everything to
  Metal, so the thread count mostly affects the CPU-side work.

## Expected failures inside the grid

- A quantised V cache (`-ctv q8_0`) generally requires flash attention in
  llama.cpp. The `fa=off, kv=q8_0` cells of the grid are therefore expected to
  fail and are recorded as `*.failed.json`, counted in the "configs failed"
  column and never used for numbers. They are kept in the grid rather than
  skipped so the failure is observable data, not an assumption.
- The 8192-token prompt (and 16384 in the context curve) may exhaust memory
  for the 3B/4B models on 8 GB. Those points are recorded as failed and the
  run continues; the summary lists them.

## Smoke test (pipeline check only)

Run on 2026-10-07 with `SMOKE=1` (one configuration, `-p 128 -n 32 -r 1` for
the sweep, `-p 128 -n 16 -r 1` for the context script, `p=128, gen=16, 1 rep`
for MLX) against the 1.5B GGUF and its MLX 4-bit equivalent. All five stages
(`fetch.sh`, `sweep.sh`, `context_length.sh`, `mlx_bench.py`, `summarize.py`)
completed and `summarize.py` produced all three table types including the
llama.cpp-vs-MLX comparison. The output lives under `results/smoke/`, which
is gitignored; its numbers are intentionally not reproduced here.

MLX worked on this chip: `mlx_lm.load` and `stream_generate` ran on the GPU
(`Device(gpu, 0)`) with no errors. The deprecation warning
`mx.metal.device_info is deprecated ... Use mx.device_info instead` was seen
once while probing; the script uses `mx.device_info()`.

## Build-time issues encountered

- Bash 3.2 treats an empty array as unbound under `set -u`; optional
  argument arrays use the `${arr[@]+"${arr[@]}"}` idiom.
- `pip install mlx-lm` hit transient DNS failures for `files.pythonhosted.org`
  and resumed on retry; the install completed.
- `shellcheck` is not installed on the target machine, so shell linting runs
  only in CI.


## 2026-10-07: first sweep discarded

The first llama.cpp sweep (1.5B model, 96-configuration grid) was run on a laptop
that was on battery, fell to 5% charge, and slept or woke more than a thousand
times over the host's lifetime. One configuration sat suspended for 1 h 44 min
before it was killed by hand, and other CPU-heavy work ran on the machine during
the sweep. Those results were moved out of `results/` into a git-ignored
quarantine directory and are not published, summarized or quoted. The scripts
were then changed to refuse such conditions and to record them (see
"Measurement conditions" in the README). The sweep has to be redone on mains
power with nothing else running.

## GitHub runner probe (2026-10-10)

Runs https://github.com/joeyvictorino/edge-inference-bench/actions/runs/38033886530
and https://github.com/joeyvictorino/edge-inference-bench/actions/runs/38035898494
(temporary branches, since deleted) ran host commands and one llama-bench
configuration on three GitHub-hosted arm64 macOS runners. Its throughput
output was printed as a Markdown table, not saved as raw JSON, so no number
from it is published; it was used only to learn the runner and to size the
grid.

| Item | `macos-15` | `macos-26` and `macos-latest` |
|---|---|---|
| Image | `macos-15-arm64` 20260907.0337.1 | `macos-26-arm64` |
| `machdep.cpu.brand_string` | `Apple M1 (Virtual)` | `Apple M1 (Virtual)` |
| `hw.model` / `hw.ncpu` / `hw.memsize` | `VirtualMac2,1` / 3 / 7516192768 | same |
| `hw.perflevel0.physicalcpu` / `hw.perflevel1.physicalcpu` | 3 / key absent (prints nothing) | not checked |
| `kern.sleeptime` | `{ sec = 0, usec = 0 } Thu Jan  1 00:00:00 1970` | not checked |
| macOS | 15.7.9 (24G830) | 26.6.2 |
| `pmset -g batt` | `Now drawing from 'AC Power'`, no battery | same |
| CPU idle samples (`top`, before the benchmark) | 38% and 2% | 28%, 57%, 58% and 5% |
| CPU idle samples (`top`, after the benchmark) | 62% and 73% | 0%, 25%, 0% and 2% |
| Metal device (llama.cpp) | `MTL0 (Apple Paravirtual device)`, `MTLGPUFamilyApple5` | same |
| Metal capabilities (llama.cpp, `macos-15`) | simdgroup reduction `false`, simdgroup matrix multiply `false`, bfloat `false`, recommended working set 5010.80 MB | not checked |
| llama.cpp (Homebrew) | formula 0.4.0, ggml 0.23.0; this `llama-bench` rejects `--version` | same |
| Metal library compile per process | about 40 s | about 40 s |
| One full-grid configuration (`-p 512,2048,8192 -n 128 -r 3`) | 22 min 18 s | 22 min 4 s and 24 min 56 s |

Consequences, applied in `.github/workflows/bench.yml` and documented in the
README: the idle threshold cannot be met on these runners, so runner mode
records the level instead of refusing; and the grid is reduced so that one
model fits in a job.

## First GitHub runner bench (2026-10-10)

- Run 38035657893 stopped in the install step: Homebrew llama.cpp 0.4.0's
  `llama-bench` rejects `--version` and the workflow grepped its output.
- Run 38036114974 stopped before the first configuration: `jq --argjson`
  received empty strings (no `hw.perflevel1.physicalcpu` key and no battery
  line on the VM).
- Run 38036322821 completed all four jobs (two models, two replicas each).
  Its raw files are under `results/gha-runner-apple-m1-virtual-7gb-macos15.7.9/`
  (replica 1, published) and `results/replicas/.../run-38036322821-replica2/`
  (replica 2, with `compare.md` and `compare.json`).

Wall time per completed sweep configuration was up to 443 s (1.5B) and up to
1001 s (3B). Configurations that started below 85% idle, among those that
completed: 1.5B 0 of 18 (replica 1) and 4 of 18 (replica 2); 3B 2 of 18 in
each replica. The context-length stage started at 87 to 97% idle in all four
jobs.

Restricting the comparison to metrics whose configuration started at 85%
idle or more in both replicas leaves the result unchanged in kind: 1.5B 12 of
52 metrics within the published IQR (median absolute difference 17.9%), 3B 4
of 55 (29.6%). Computed with:

    python3 - <<'PY'
    import json, statistics
    H = 'gha-runner-apple-m1-virtual-7gb-macos15.7.9'
    R = 'results/replicas/%s/run-38036322821-replica2' % H
    c = json.load(open(R + '/compare.json'))
    def met(base, m, stage, item):
        p = '%s/%s/sweep/%s.env.json' % (base, m, item) if stage == 'sweep' else '%s/%s/context/stage-conditions.json' % (base, m)
        return json.load(open(p)).get('quiet_threshold_met')
    for m in sorted(c['by_model']):
        rows = [r for r in c['rows'] if r['model'] == m
                and met('results/' + H, m, r['stage'], r['item']) and met(R, m, r['stage'], r['item'])]
        print(m, len(rows), sum(r['within_published_iqr'] for r in rows),
              round(statistics.median(abs(r['delta_pct']) for r in rows), 1))
    PY

The idle check sees only the guest's CPU. Contention on the physical host or
its GPU is invisible to it; that is a plausible cause of the spread, not a
verified one.

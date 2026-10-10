"""sweep.sh must refuse battery, kill hung runs, and never count a run that
the machine slept through. These tests run the real script against stub
llama-bench and pmset programs; nothing here measures anything."""
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, "scripts", "sweep.sh")

LLAMA_STUB = """#!/usr/bin/env bash
case "$1" in --version) echo "version: stub (build 0, commit 0000000)"; exit 0;; esac
case "${STUB_MODE:-ok}" in
  hang) sleep 30 ;;
  *) echo '[{"model_filename":"stub.gguf","model_type":"stub","model_size":1,"n_prompt":512,"n_gen":0,"avg_ts":100.0,"stddev_ts":1.0,"samples_ts":[99.0,100.0,101.0]}]' ;;
esac
"""
PMSET_STUB = """#!/usr/bin/env bash
if [ "${STUB_POWER:-ac}" = "ac" ]; then
  echo "Now drawing from 'AC Power'"; echo " -InternalBattery-0 (id=1)	80%; charging; 1:00 remaining present: true"
else
  echo "Now drawing from 'Battery Power'"; echo " -InternalBattery-0 (id=1)	5%; discharging; (no estimate) present: true"
fi
"""
# prints a different value on every call, as if the machine slept between them
TICK_STUB = """#!/usr/bin/env bash
f="$TICK_FILE"; n=$(cat "$f" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "$f"; echo "tick-$n"
"""


def write_exe(path, body):
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


class SweepConditions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.bin = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin)
        write_exe(os.path.join(self.bin, "llama-bench"), LLAMA_STUB)
        write_exe(os.path.join(self.bin, "pmset"), PMSET_STUB)
        self.tick = os.path.join(self.tmp, "tick.sh")
        write_exe(self.tick, TICK_STUB)
        self.model = os.path.join(self.tmp, "stub.gguf")
        with open(self.model, "wb") as f:
            f.write(b"not a model")
        self.out = os.path.join(self.tmp, "out")

    def run_sweep(self, **env):
        # The real sleep stamp is pinned to a constant: a laptop that really
        # sleeps during a 2-second test must not change the outcome. Tests that
        # want a sleep pass ASSAY_SLEEP_STAMP_CMD themselves.
        e = dict(os.environ, PATH=self.bin + os.pathsep + os.environ["PATH"], SMOKE="1",
                 TICK_FILE=os.path.join(self.tmp, "tick.n"), ASSAY_SLEEP_STAMP_CMD="echo constant", ASSAY_CPU_IDLE_CMD="echo 99")
        e.update(env)
        return subprocess.run(["bash", SWEEP, self.model, self.out], env=e,
                              capture_output=True, text=True, timeout=120)

    def env_of(self, name):
        with open(os.path.join(self.out, name + ".env.json")) as f:
            return json.load(f)

    def files(self):
        return sorted(os.listdir(self.out)) if os.path.isdir(self.out) else []

    def test_refuses_battery(self):
        r = self.run_sweep(STUB_POWER="battery")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("refusing to benchmark on battery", r.stderr)
        self.assertEqual([f for f in self.files() if f.endswith(".json") and f != "manifest.json"], [])

    def test_clean_run_is_counted_and_conditions_recorded(self):
        r = self.run_sweep()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = [f for f in self.files() if f.startswith("t4_b512_ub256") and f.endswith("kvf16.json")]
        self.assertEqual(len(res), 1, self.files())
        env = self.env_of("t4_b512_ub256_fa1_kvf16")
        self.assertEqual(env["power_before"]["source"], "AC")
        self.assertFalse(env["slept_during_run"])
        self.assertFalse(env["timed_out"])

    def test_battery_override_records_battery(self):
        r = self.run_sweep(STUB_POWER="battery", ALLOW_BATTERY="1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        env = self.env_of("t4_b512_ub256_fa1_kvf16")
        self.assertEqual(env["power_before"]["source"], "Battery")

    def test_hung_config_is_killed_and_recorded(self):
        r = self.run_sweep(STUB_MODE="hang", CONFIG_TIMEOUT="2")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("t4_b512_ub256_fa1_kvf16.failed.json", self.files())
        env = self.env_of("t4_b512_ub256_fa1_kvf16")
        self.assertTrue(env["timed_out"])
        self.assertLess(env["wall_seconds"], 20)

    def test_run_through_a_sleep_is_never_counted(self):
        r = self.run_sweep(ASSAY_SLEEP_STAMP_CMD=self.tick)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("interrupted by system sleep", r.stdout)
        self.assertIn("t4_b512_ub256_fa1_kvf16.interrupted.json", self.files())
        self.assertNotIn("t4_b512_ub256_fa1_kvf16.json", self.files())
        env = self.env_of("t4_b512_ub256_fa1_kvf16")
        self.assertTrue(env["slept_during_run"])
        # a later run on a quiet machine retries it and replaces the interrupted file
        r2 = self.run_sweep()
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        self.assertIn("t4_b512_ub256_fa1_kvf16.json", self.files())
        self.assertNotIn("t4_b512_ub256_fa1_kvf16.interrupted.json", self.files())

    def test_busy_machine_blocks_a_configuration_and_it_is_retried(self):
        r = self.run_sweep(ASSAY_CPU_IDLE_CMD="echo 40", QUIET_TIMEOUT="2", QUIET_POLL="1")
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("machine not idle", r.stdout)
        self.assertNotIn("t4_b512_ub256_fa1_kvf16.json", self.files())
        self.assertNotIn("t4_b512_ub256_fa1_kvf16.env.json", self.files())
        r2 = self.run_sweep()
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        self.assertIn("t4_b512_ub256_fa1_kvf16.json", self.files())

    def test_idle_level_is_recorded(self):
        r = self.run_sweep(ASSAY_CPU_IDLE_CMD="echo 93.5")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.env_of("t4_b512_ub256_fa1_kvf16")["cpu_idle_before_pct"], 93.5)

    def test_busy_local_machine_is_still_refused_with_runner_mode_unset(self):
        r = self.run_sweep(ASSAY_CPU_IDLE_CMD="echo 40", QUIET_TIMEOUT="1", QUIET_POLL="1")
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)

    def test_busy_github_runner_runs_and_records_the_threshold_miss(self):
        r = self.run_sweep(ASSAY_CPU_IDLE_CMD="echo 41.5", QUIET_TIMEOUT="1", QUIET_POLL="1", BENCH_HOST="gha-macos")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("never reached 85% idle", r.stdout)
        env = self.env_of("t4_b512_ub256_fa1_kvf16")
        self.assertEqual(env["cpu_idle_before_pct"], 41.5)
        self.assertFalse(env["quiet_threshold_met"])
        self.assertEqual(env["quiet_threshold_pct"], 85)
        with open(os.path.join(self.out, "manifest.json")) as f:
            man = json.load(f)
        self.assertEqual(man["host"]["host_kind"], "gha-macos")
        self.assertTrue(man["host"]["runner"]["virtualized"])

    def test_quiet_run_records_threshold_met(self):
        r = self.run_sweep()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(self.env_of("t4_b512_ub256_fa1_kvf16")["quiet_threshold_met"])

    def test_machine_that_quiets_down_is_waited_for(self):
        # busy on the first sample, quiet on the second
        flip = os.path.join(self.tmp, "flip.sh")
        write_exe(flip, "#!/usr/bin/env bash\nf=\"$TICK_FILE.idle\"; n=$(cat \"$f\" 2>/dev/null || echo 0); echo $((n+1)) > \"$f\"; [ \"$n\" -ge 1 ] && echo 96 || echo 30\n")
        r = self.run_sweep(ASSAY_CPU_IDLE_CMD=flip, QUIET_POLL="1", QUIET_TIMEOUT="10")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.env_of("t4_b512_ub256_fa1_kvf16")["cpu_idle_before_pct"], 96)


class PowerHelpers(unittest.TestCase):
    """power_source must not be fooled by SIGPIPE under pipefail."""

    def test_power_source_survives_a_chatty_pmset_under_pipefail(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            os.makedirs(bin_dir)
            # first line says AC; then a lot more output, so a reader that
            # stops after one line would make this process die of SIGPIPE
            write_exe(os.path.join(bin_dir, "pmset"),
                      "#!/usr/bin/env bash\necho \"Now drawing from 'AC Power'\"\nfor i in $(seq 1 5000); do echo \"line $i 55%\"; done\n")
            script = "set -euo pipefail; source scripts/lib/host.sh; for i in 1 2 3 4 5 6 7 8 9 10; do power_source; done; battery_pct"
            r = subprocess.run(["bash", "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60,
                               env=dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"]))
            self.assertEqual(r.returncode, 0, r.stderr)
            lines = r.stdout.split()
            self.assertEqual(lines[:10], ["AC"] * 10, r.stdout)
            self.assertEqual(lines[10], "55")

    def test_battery_is_reported_as_battery(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            os.makedirs(bin_dir)
            write_exe(os.path.join(bin_dir, "pmset"),
                      "#!/usr/bin/env bash\necho \"Now drawing from 'Battery Power'\"\necho \" -InternalBattery-0\t5%; discharging\"\n")
            r = subprocess.run(["bash", "-c", "set -euo pipefail; source scripts/lib/host.sh; power_source; battery_pct"],
                               cwd=ROOT, capture_output=True, text=True, timeout=60,
                               env=dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"]))
            self.assertEqual(r.stdout.split(), ["Battery", "5"], r.stdout + r.stderr)


class HostKind(unittest.TestCase):
    """A GitHub runner must never be filed under the operator's own host."""

    def sh(self, script, **env):
        e = dict(os.environ)
        e.pop("BENCH_HOST", None)
        e.update(env)
        return subprocess.run(["bash", "-c", "set -euo pipefail; source scripts/lib/host.sh; " + script],
                              cwd=ROOT, capture_output=True, text=True, timeout=60, env=e)

    def test_default_is_local_with_no_runner_record(self):
        r = self.sh("host_kind; runner_json; host_slug")
        self.assertEqual(r.returncode, 0, r.stderr)
        kind, runner, slug = r.stdout.split()
        self.assertEqual(kind, "local")
        self.assertEqual(runner, "null")
        self.assertFalse(slug.startswith("gha-runner-"), slug)

    def test_gha_runner_is_prefixed_and_records_the_run(self):
        r = self.sh("host_slug; runner_json", BENCH_HOST="gha-macos", GITHUB_RUN_ID="42",
                    GITHUB_REPOSITORY="o/r", GITHUB_SERVER_URL="https://github.com", ImageOS="macos15",
                    ImageVersion="20261001.1", RUNNER_ARCH="ARM64", BENCH_RUNNER_LABEL="macos-15")
        self.assertEqual(r.returncode, 0, r.stderr)
        slug, rest = r.stdout.split("\n", 1)
        self.assertTrue(slug.startswith("gha-runner-"), slug)
        runner = json.loads(rest)
        self.assertTrue(runner["virtualized"])
        self.assertEqual(runner["run_url"], "https://github.com/o/r/actions/runs/42/attempts/1")
        self.assertEqual(runner["runs_on"], "macos-15")

    def test_unknown_host_kind_is_refused(self):
        r = self.sh("host_slug", BENCH_HOST="laptop")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown BENCH_HOST", r.stderr)


class LlamaVersion(unittest.TestCase):
    """Older llama-bench builds reject --version; the manifest must still carry a version."""

    def test_falls_back_to_llama_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            os.makedirs(bin_dir)
            write_exe(os.path.join(bin_dir, "llama-bench"),
                      "#!/usr/bin/env bash\necho 'usage: llama-bench [options]'\necho 'error: invalid parameter for argument: --version' >&2\nexit 1\n")
            write_exe(os.path.join(bin_dir, "llama-cli"),
                      "#!/usr/bin/env bash\necho 'version: 9999 (abc1234)'\necho 'built with stub for arm64'\n")
            r = subprocess.run(["bash", "-c", "set -euo pipefail; source scripts/lib/host.sh; llama_cache_version; "
                                "llama_version_string; llama_build_line"],
                               cwd=ROOT, capture_output=True, text=True, timeout=60,
                               env=dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"]))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.splitlines(), ["9999 (abc1234)", "built with stub for arm64"])


if __name__ == "__main__":
    unittest.main()

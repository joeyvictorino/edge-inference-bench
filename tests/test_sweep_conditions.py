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
                 TICK_FILE=os.path.join(self.tmp, "tick.n"), ASSAY_SLEEP_STAMP_CMD="echo constant")
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


if __name__ == "__main__":
    unittest.main()

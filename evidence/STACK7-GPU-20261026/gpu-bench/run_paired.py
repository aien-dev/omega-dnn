# THROWAWAY paired A/B runner (lane OmegaDnn.GpuDnn, 2026-10-26), not product code.
# Protocol frozen in /home/drakestapleton/handoffs/OmegaDnn/GPU.md section 2 before the first timed run.
# Usage: python run_paired.py RUNDIR [ROUNDS=5] [WARMUP_ROUNDS=1]
# Each round runs both arms, one process each, in an order from random.Random(20261026); every arm process
# holds /tmp/aien-gb10.lock only for its own lifetime (flock around the GPU command alone). After every arm
# process: loadavg and `nvidia-smi --query-compute-apps=pid` are recorded.
import json, os, random, subprocess, sys, time
rundir = sys.argv[1]; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5; wr = int(sys.argv[3]) if len(sys.argv) > 3 else 1
B = "/tmp/OmegaDnn/gpu/bench"; PY = "/home/drakestapleton/.venv/bin/python"
os.makedirs(rundir, exist_ok=True)
rng = random.Random(20261026)
log = open(f"{rundir}/rounds.jsonl", "a")
for r in range(-wr, rounds):
    order = ["omega", "torch"]; rng.shuffle(order)
    for arm in order:
        d = f"{rundir}/r{r}/{arm}"; os.makedirs(d, exist_ok=True)
        if arm == "omega": cmd = [f"{B}/omega_dnn_bench", d]
        else: cmd = [PY, f"{B}/torch_arm.py", f"{rundir}/inputs", d]
        load0 = open("/proc/loadavg").read().split()[:3]
        t0 = time.time()
        p = subprocess.run(["flock", "-w", "900", "/tmp/aien-gb10.lock", f"{B}/held.sh", "taskset", "-c", "8-9", *cmd], capture_output=True, text=True)
        t1 = time.time()
        open(f"{d}/stdout.jsonl", "w").write(p.stdout); open(f"{d}/stderr.txt", "w").write(p.stderr)
        held = [l for l in p.stderr.splitlines() if l.startswith("LOCK_HELD_S ")]
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        rec = {"round": r, "arm": arm, "rc": p.returncode, "wall_s": round(t1 - t0, 3), "lock_held_s": held[-1].split()[1] if held else None,
               "load_before": load0, "load_after": open("/proc/loadavg").read().split()[:3], "compute_apps_after": apps}
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if p.returncode != 0: sys.exit(f"arm {arm} round {r} failed rc={p.returncode}")

#!/bin/sh
# gpu_run.sh LABEL CMD...: BOARD start line, flock around CMD only, BOARD end line, compute-apps check.
# Output of CMD goes to stdout; summary line to stderr.
B=/home/drakestapleton/handoffs/BOARD.md
label=$1; shift
echo "$(date -u +%FT%TZ) OmegaDnn.GpuDnn: START $label 0 running (lock /tmp/aien-gb10.lock, driver 580.173.02)" >> $B
t0=$(date +%s)
flock -w 900 /tmp/aien-gb10.lock /tmp/OmegaDnn/gpu/bench/held.sh "$@" 2>/tmp/OmegaDnn/gpu/.held.$$; rc=$?
cat /tmp/OmegaDnn/gpu/.held.$$ >&2
held=$(grep '^LOCK_HELD_S' /tmp/OmegaDnn/gpu/.held.$$ | tail -1 | cut -d' ' -f2); rm -f /tmp/OmegaDnn/gpu/.held.$$
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
napps=$(printf '%s' "$apps" | grep -c .)
echo "$(date -u +%FT%TZ) OmegaDnn.GpuDnn: END $label lock_held_s=$held rc=$rc compute_apps_rows=$napps" >> $B
echo "GPU_RUN $label rc=$rc lock_held_s=$held compute_apps_rows=$napps apps=[$apps] date=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)" >&2
exit $rc

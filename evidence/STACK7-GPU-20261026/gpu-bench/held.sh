#!/bin/sh
# Runs under flock: reports how long the lock was held (acquire to command exit) on stderr.
t0=$(date +%s.%N)
"$@"; rc=$?
t1=$(date +%s.%N)
echo "LOCK_HELD_S $(echo "$t1 - $t0" | bc)" >&2
exit $rc

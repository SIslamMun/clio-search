#!/usr/bin/env python3
"""Run a Python script with /proc/self/io measurement around it.

Usage: io_wrap.py <script.py> [args...]

Prints IO_DELTA: {...} as the last line — total bytes/syscalls the wrapped
script performed. Works inside apptainer because /proc/self/io is a kernel
interface, not a binary dependency.

Why this and not strace: strace counts overall process syscalls including
opencode's Bun runtime startup (~16k opens before the agent does anything).
This wrapper measures only the wrapped script's I/O — which is what we
actually want to compare across A/B/C arms.
"""
import json
import runpy
import sys


def read_proc_io():
    out = {}
    for line in open("/proc/self/io"):
        k, v = line.split(":", 1)
        out[k.strip()] = int(v.strip())
    return out


if len(sys.argv) < 2:
    print("usage: io_wrap.py <script.py> [args...]", file=sys.stderr)
    sys.exit(2)

io_start = read_proc_io()
script = sys.argv[1]
sys.argv = sys.argv[1:]  # script sees its own argv0 = sys.argv[0]
exit_code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    exit_code = (e.code if isinstance(e.code, int) else 1) or 0
except Exception as e:
    print(f"io_wrap: wrapped script raised: {e!r}", file=sys.stderr)
    exit_code = 1

io_end = read_proc_io()
delta = {k: io_end[k] - io_start[k] for k in io_start}

# Print the I/O delta as the last line so our parser can find it.
# Fields meaning (from /proc/[pid]/io):
#   rchar       — bytes read (across all reads, including cached)
#   wchar       — bytes written
#   syscr       — read-like syscall count
#   syscw       — write-like syscall count
#   read_bytes  — bytes physically fetched from disk
#   write_bytes — bytes physically pushed to disk
#   cancelled_write_bytes — pages dirtied then truncated before flush
print("IO_DELTA: " + json.dumps(delta))
sys.exit(exit_code)

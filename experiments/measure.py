"""Run a Python script and report wall time and peak memory (Windows peak working set).

Usage: python measure.py <workdir> <script.py>
"""
import os
import runpy
import sys
import time

import psutil

workdir, script = sys.argv[1], sys.argv[2]
os.chdir(workdir)
sys.path.insert(0, workdir)
sys.argv = [script]

proc = psutil.Process()
t0 = time.perf_counter()
try:
    runpy.run_path(script, run_name="__main__")
finally:
    elapsed = time.perf_counter() - t0
    mem = proc.memory_info()
    peak = getattr(mem, "peak_wset", mem.rss)
    print(f"[measure] wall time: {elapsed:.2f} s")
    print(f"[measure] peak memory: {peak / 2**20:.1f} MiB")

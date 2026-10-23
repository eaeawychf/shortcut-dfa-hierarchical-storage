"""Elisabeth-b4: measure ONE sub-table (2^24 entries) per organisation and extrapolate.

Building all 90 tables is avoided on purpose; per-table cost is linear, so totals are extrapolated.
"""
import json
import time

import numpy as np
import psutil

import table_decomposer as td

h, n = td.CIPHERS["Eb4"]
size = 16 ** n
res = {"cipher": "Eb4", "table_entries": size}


def rss():
    return psutil.Process().memory_info().rss / 2**20


t = time.perf_counter()
res["global_negation_holds"] = td.has_global_negation(h, n)
res["negation_check_s"] = time.perf_counter() - t

# dense T_{1,0} and T_{15,0}
t = time.perf_counter()
T1 = td.build_table(h, n, 1, 0)
res["build_one_table_s"] = time.perf_counter() - t
T15 = td.build_table(h, n, 15, 0)

# S2: rebuild T_{15,0} from T_{1,0}
t = time.perf_counter()
src = td.shift_index(np.arange(size, dtype=np.uint32), n, 0, -1)
T15_rec = ((-T1[src].astype(np.int16)) & 15).astype(np.uint8)
res["rebuild_from_symmetry_s"] = time.perf_counter() - t
res["S2_rebuild_matches"] = bool(np.array_equal(T15, T15_rec))
del src, T15_rec

# lookup cost per query
dh = 7
t = time.perf_counter()
for _ in range(10):
    hits = np.flatnonzero(T15 == dh)
res["dense_scan_lookup_s"] = (time.perf_counter() - t) / 10
res["hits_per_query"] = int(hits.size)

t = time.perf_counter()
order, offsets = td._invert(T15, n)
res["invert_one_table_s"] = time.perf_counter() - t
t = time.perf_counter()
for _ in range(1000):
    hits2 = order[offsets[dh]:offsets[dh + 1]]
res["inverted_lookup_s"] = (time.perf_counter() - t) / 1000
res["inverted_matches_dense"] = bool(np.array_equal(hits, hits2))

dense_mib = T15.nbytes / 2**20
inv_mib = (order.nbytes + offsets.nbytes) / 2**20
res["one_table_dense_MiB"] = dense_mib
res["one_table_inverted_MiB"] = inv_mib
res["peak_MiB"] = getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2**20

# extrapolation (uint8 per entry; the original text format is ~14 bytes/line)
res["extrapolated"] = {
    "uniform_all_15x6_dense_MiB": 90 * dense_mib,
    "symmetric_8x6_dense_MiB": 48 * dense_mib,
    "inverted_all_15x6_MiB": 90 * inv_mib,
    "attack_needs_only_known_delta_6_tables_dense_MiB": 6 * dense_mib,
    "original_text_format_all_90_GiB_approx": 90 * size * 14 / 2**30,
    "uniform_build_all_s": 90 * res["build_one_table_s"],
    "symmetric_build_all_s": 48 * res["build_one_table_s"],
}
print(json.dumps(res, indent=1))

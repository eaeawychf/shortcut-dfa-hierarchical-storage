"""Elisabeth-4 key filtering with a pluggable filter-table layer.

The merge loop, intersect() and determine_inter() are copied unchanged from
DFA_Elisabeth_family/Elisabeth-4_filtering_key.py; only "how T is stored and queried" changes.

Usage: python attack_harness.py <store> [--queries-only] [--fast-intersect]
  store: original | uniform | inverted | symmetric | symmetric_inverted
"""
import copy
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import psutil

import table_decomposer as td

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DFA_Elisabeth_family")


def peak_mib():
    m = psutil.Process().memory_info()
    return getattr(m, "peak_wset", m.rss) / 2**20


# ---------------------------------------------------------------- copied from the original script
def read_truth_table(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    truth_table = []
    for line in lines:
        line = line.rstrip("\n")
        new_list = line.split(" ")
        row = [int(i) for i in new_list]
        truth_table.append(row)
    return truth_table


def read_simulate_result(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    time_infs, diff_val_infs, pos_infs, white_vec_infs = [], [], [], []
    for line in lines:
        line = line.rstrip("\n")
        tmp_pair = line.split("\t")
        time_infs.append(int(tmp_pair[0]))
        diff_val_infs.append(int(tmp_pair[3]))
        pos_infs.append([int(x) for x in tmp_pair[1].strip('[').strip(']').split(',')])
        white_vec_infs.append([int(x) for x in tmp_pair[2].strip('[').strip(']').split(',')])
    return time_infs, diff_val_infs, pos_infs, white_vec_infs


def determine_inter(index_pair, sol1, sol2):
    for pair in index_pair:
        if sol1[pair[0]] == sol2[pair[1]]:
            continue
        else:
            return 0
    return 1


def intersect(index1, index2, solution1, solution2):
    add_index = []
    cor_index = []
    cor_index_pair = []
    for i in range(len(index2)):
        if index2[i] in index1:
            cor_index_pair.append((index1.index(index2[i]), i))
        else:
            add_index.append(index2[i])
            cor_index.append(i)
    new_index = index1 + add_index
    new_solution = []
    if len(add_index) == 0:
        for tmp1 in solution1:
            for tmp2 in solution2:
                if determine_inter(cor_index_pair, tmp1, tmp2):
                    new_solution.append(tmp1)
    else:
        for tmp1 in solution1:
            for tmp2 in solution2:
                if determine_inter(cor_index_pair, tmp1, tmp2):
                    new_sol = copy.deepcopy(tmp1)
                    for i in cor_index:
                        new_sol.append(tmp2[i])
                    new_solution.append(new_sol)
    return new_index, new_solution


def intersect_hash(index1, index2, solution1, solution2):
    """Same output (and order) as intersect(), but a hash join instead of the |S1|*|S2| double loop."""
    add_index = []
    cor_index = []
    cor_index_pair = []
    for i in range(len(index2)):
        if index2[i] in index1:
            cor_index_pair.append((index1.index(index2[i]), i))
        else:
            add_index.append(index2[i])
            cor_index.append(i)
    new_index = index1 + add_index
    buckets = {}
    for tmp2 in solution2:
        buckets.setdefault(tuple(tmp2[p[1]] for p in cor_index_pair), []).append(tmp2)
    new_solution = []
    for tmp1 in solution1:
        matches = buckets.get(tuple(tmp1[p[0]] for p in cor_index_pair))
        if not matches:
            continue
        if len(add_index) == 0:
            new_solution.extend(tmp1 for _ in matches)
        else:
            for tmp2 in matches:
                new_solution.append(tmp1 + [tmp2[i] for i in cor_index])
    return new_index, new_solution


def read_path(file_name):
    f = open(file_name, "r")
    return [int(line.rstrip("\n").split(" ")[0]) for line in f.readlines()]


TEST_KEY = list(range(16)) * 16
FAULT_P = 0
DELTA = 15  # the fault value; known to the attacker (see the original script's comment)


# ---------------------------------------------------------------- pluggable table layer
class OriginalTables:
    """Exactly what the original script does: 4 text tables as lists of rows, linear scan."""

    def __init__(self):
        self.t = [read_truth_table(os.path.join(REPO, "E4_Diff_tables", "New_Small_h_x%s_v%s.txt" % (j, DELTA)))
                  for j in range(4)]

    def solutions(self, choice, dif_value, w_vec):
        return [[(tmp[i] - w_vec[i]) % 16 for i in range(4)] for tmp in self.t[choice] if tmp[-1] == dif_value]


class StoreTables:
    def __init__(self, name):
        self.store = td.STORES[name]("E4")

    def solutions(self, choice, dif_value, w_vec):
        idx = self.store.lookup(DELTA, choice, dif_value)
        x = np.stack(td.unpack(idx, 4), axis=1).astype(np.int16)
        return ((x - np.asarray(w_vec, dtype=np.int16)) % 16).tolist()


def main():
    name = sys.argv[1]
    queries_only = "--queries-only" in sys.argv
    fast = "--fast-intersect" in sys.argv
    join = intersect_hash if fast else intersect
    sim_times, sim_diff_vals, sim_poss, sim_w_vecs = read_simulate_result(
        os.path.join(REPO, "tmp_result_file", "E4_useful_dif_information_3000_0.txt"))
    merge_path = read_path(os.path.join(REPO, "useful_result", "E4_merge_path_3000_0_(114, 166).txt"))
    res = {"store": name, "intersect": "hash" if fast else "original"}

    base_mib = psutil.Process().memory_info().rss / 2**20
    t = time.perf_counter()
    tables = OriginalTables() if name == "original" else StoreTables(name)
    res["build_s"] = time.perf_counter() - t
    res["rss_after_build_mib"] = psutil.Process().memory_info().rss / 2**20
    res["table_rss_delta_mib"] = res["rss_after_build_mib"] - base_mib
    if name != "original":
        res["table_nbytes_before_queries"] = tables.store.nbytes()

    def query(point):
        k = sim_times.index(point)
        pos = sim_poss[k]
        return pos, tables.solutions(pos.index(FAULT_P), sim_diff_vals[k], sim_w_vecs[k])

    # pass 1: every query on the merge path, timed on its own, hashed for the equivalence check
    digest = hashlib.sha256()
    t = time.perf_counter()
    for p in merge_path:
        _, sol = query(p)
        digest.update(json.dumps(sol).encode())
    res["query_all_s"] = time.perf_counter() - t
    res["n_queries"] = len(merge_path)
    res["query_digest"] = digest.hexdigest()[:16]
    if name != "original":
        res["table_nbytes_after_queries"] = tables.store.nbytes()

    if not queries_only:
        # pass 2: the original merge loop
        q_time = 0.0
        inter_times = 0
        t0 = time.perf_counter()
        t = time.perf_counter()
        tmp_pos, tmp_solution_space = query(merge_path[0])
        q_time += time.perf_counter() - t
        steps = 0
        for i in range(1, len(merge_path)):
            t = time.perf_counter()
            next_pos, next_solution_space = query(merge_path[i])
            q_time += time.perf_counter() - t
            inter_times += len(tmp_solution_space) * len(next_solution_space)
            tmp_pos, tmp_solution_space = join(tmp_pos, next_pos, tmp_solution_space, next_solution_space)
            steps = i
            print("step %d point %d keys %d candidates %d elapsed %.1fs peak %.0fMiB"
                  % (i, merge_path[i], len(tmp_pos), len(tmp_solution_space), time.perf_counter() - t0, peak_mib()),
                  file=sys.stderr, flush=True)
            if len(tmp_solution_space) > 2**22:
                print("Too huge space!")
                break
            if len(tmp_solution_space) < 1 or (len(tmp_solution_space) == 1 and len(tmp_solution_space[0]) == 256):
                break
        res["attack_total_s"] = time.perf_counter() - t0
        res["attack_query_s"] = q_time
        res["merge_steps"] = steps
        res["log2_intersections"] = math.log(inter_times, 2)
        res["final_candidates"] = len(tmp_solution_space)
        res["recovered_key_positions"] = len(tmp_pos)
        res["right_key_found"] = any(all(v[i] == TEST_KEY[tmp_pos[i]] for i in range(len(tmp_pos)))
                                     for v in tmp_solution_space)
    res["peak_mib"] = peak_mib()
    print(json.dumps(res))


if __name__ == "__main__":
    main()

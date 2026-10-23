"""Check intersect_hash() == intersect() (same lists, same order) on the first merge steps of E-4."""
import sys
import attack_harness as ah

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
sim_times, sim_diff_vals, sim_poss, sim_w_vecs = ah.read_simulate_result(
    ah.os.path.join(ah.REPO, "tmp_result_file", "E4_useful_dif_information_3000_0.txt"))
path = ah.read_path(ah.os.path.join(ah.REPO, "useful_result", "E4_merge_path_3000_0_(114, 166).txt"))
tables = ah.OriginalTables()

def q(p):
    k = sim_times.index(p)
    return sim_poss[k], tables.solutions(sim_poss[k].index(ah.FAULT_P), sim_diff_vals[k], sim_w_vecs[k])

a_pos, a = q(path[0]); b_pos, b = a_pos, a
for i in range(1, steps + 1):
    np_, ns = q(path[i])
    a_pos, a = ah.intersect(a_pos, np_, a, ns)
    b_pos, b = ah.intersect_hash(b_pos, np_, b, ns)
    print(i, len(a), a_pos == b_pos and a == b, flush=True)

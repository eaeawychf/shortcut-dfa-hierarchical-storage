"""Check the NLUT-induced symmetries that would let T_{delta,j} be stored more compactly.

S1: h(x + 8*(1,..,1)) == -h(x)                     (negacyclic: S[x+8] = -S[x])
S2: T_{16-d,j}(x) == -T_{d,j}(x - d*e_j)           (holds for any h, by definition of dh)
"""
import random
import sys

sys.path.insert(0, r"C:\Users\91828\Desktop\fushi_safety\DFA_Elisabeth_family")

S = [
    [3, 2, 6, 12, 10, 0, 1, 11, 13, 14, 10, 4, 6, 0, 15, 5],
    [4, 11, 4, 4, 4, 15, 9, 12, 12, 5, 12, 12, 12, 1, 7, 4],
    [11, 10, 12, 2, 2, 11, 13, 14, 5, 6, 4, 14, 14, 5, 3, 2],
    [5, 9, 13, 2, 11, 10, 12, 5, 11, 7, 3, 14, 5, 6, 4, 11],
    [3, 0, 11, 8, 13, 14, 13, 11, 13, 0, 5, 8, 3, 2, 3, 5],
    [8, 13, 12, 12, 3, 15, 12, 7, 8, 3, 4, 4, 13, 1, 4, 9],
    [4, 2, 9, 13, 10, 12, 10, 7, 12, 14, 7, 3, 6, 4, 6, 9],
    [10, 2, 5, 5, 3, 13, 15, 1, 6, 14, 11, 11, 13, 3, 1, 15],
]


def h4(x1, x2, x3, x4):
    a, b, c, d = (x1 + x2) & 15, (x2 + x3) & 15, (x3 + x4) & 15, (x4 + x1) & 15
    s1a, s2b, s3c, s4d = S[0][a], S[1][b], S[2][c], S[3][d]
    return (S[4][(x1 + s2b + s3c) & 15] + S[5][(x2 + s3c + s4d) & 15]
            + S[6][(x3 + s4d + s1a) & 15] + S[7][(x4 + s1a + s2b) & 15]) & 15


def is_negacyclic(t):
    return all(t[i + 8] == (-t[i]) & 15 for i in range(8))


def check(h, n, samples):
    s1 = s2 = 0
    for x in samples:
        y = [(v + 8) & 15 for v in x]
        s1 += h(*y) == (-h(*x)) & 15
    for x in samples:
        d, j = random.randrange(1, 16), random.randrange(n)
        xd = list(x); xd[j] = (xd[j] + d) & 15
        xm = list(x); xm[j] = (xm[j] - d) & 15
        t_d_at_xm = (h(*xm) - h(*x)) & 15          # T_{d,j}(x - d e_j)
        t_16md = (h(*x) - h(*xm)) & 15              # T_{16-d,j}(x)
        s2 += t_16md == (-t_d_at_xm) & 15
    return s1, s2, len(samples)


if __name__ == "__main__":
    import construct_diff_tables_b4 as b4  # noqa: E402  (written alongside, see below)
    print("E-4 NLUTs negacyclic:", all(is_negacyclic(t) for t in S))
    all4 = [[a, b, c, d] for a in range(16) for b in range(16) for c in range(16) for d in range(16)]
    print("E-4 (exhaustive) S1/S2/total:", check(h4, 4, all4))
    print("E-b4 NLUTs negacyclic:", all(is_negacyclic(t) for t in b4.NLUTS))
    random.seed(1)
    samp = [[random.randrange(16) for _ in range(6)] for _ in range(200000)]
    print("E-b4 (200k samples) S1/S2/total:", check(b4.h_func_b4, 6, samp))

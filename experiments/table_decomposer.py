"""Alternative organisations of the Shortcut2Secrets filter table T = {T_{delta,j}}.

T_{delta,j}(x) = h(x) - h(x + delta*e_j) mod 16, for delta in 1..15 and j the faulted input of h.
Index convention matches the repo's text tables: x1 is the most significant nibble.

Stores (all expose lookup(delta, j, dh) -> sorted np.ndarray of input indices x with T_{delta,j}(x) == dh):
  UniformStore      : the paper's 15 x n decomposition, one dense uint8 table per (delta, j)
  InvertedStore     : same tables, but bucketed by dh (CSR) so a lookup is a slice instead of a scan
  SymmetricStore    : keeps only delta <= 8, and (if h(x+8*1) == -h(x)) only the half with x1 < 8;
                      the requested T_{delta,j} is rebuilt on the fly from the NLUT symmetries
  SymmetricInvertedStore : SymmetricStore + an inverted index cached per (delta, j) actually queried
"""
import numpy as np

# ---------------------------------------------------------------- Elisabeth-4 h (8 NLUTs)
E4_S = np.array([
    [3, 2, 6, 12, 10, 0, 1, 11, 13, 14, 10, 4, 6, 0, 15, 5],
    [4, 11, 4, 4, 4, 15, 9, 12, 12, 5, 12, 12, 12, 1, 7, 4],
    [11, 10, 12, 2, 2, 11, 13, 14, 5, 6, 4, 14, 14, 5, 3, 2],
    [5, 9, 13, 2, 11, 10, 12, 5, 11, 7, 3, 14, 5, 6, 4, 11],
    [3, 0, 11, 8, 13, 14, 13, 11, 13, 0, 5, 8, 3, 2, 3, 5],
    [8, 13, 12, 12, 3, 15, 12, 7, 8, 3, 4, 4, 13, 1, 4, 9],
    [4, 2, 9, 13, 10, 12, 10, 7, 12, 14, 7, 3, 6, 4, 6, 9],
    [10, 2, 5, 5, 3, 13, 15, 1, 6, 14, 11, 11, 13, 3, 1, 15],
], dtype=np.uint8)


def h_e4(x1, x2, x3, x4):
    S = E4_S
    s1a, s2b, s3c, s4d = S[0][(x1 + x2) & 15], S[1][(x2 + x3) & 15], S[2][(x3 + x4) & 15], S[3][(x4 + x1) & 15]
    return (S[4][(x1 + s2b + s3c) & 15] + S[5][(x2 + s3c + s4d) & 15]
            + S[6][(x3 + s4d + s1a) & 15] + S[7][(x4 + s1a + s2b) & 15]) & 15


# ---------------------------------------------------------------- Elisabeth-b4 h (18 NLUTs)
EB4_S = np.array([
    [8, 12, 12, 9, 11, 11, 8, 15, 8, 4, 4, 7, 5, 5, 8, 1],
    [11, 3, 1, 2, 12, 3, 11, 12, 5, 13, 15, 14, 4, 13, 5, 4],
    [5, 7, 8, 0, 12, 4, 10, 5, 11, 9, 8, 0, 4, 12, 6, 11],
    [12, 12, 10, 0, 13, 1, 7, 7, 4, 4, 6, 0, 3, 15, 9, 9],
    [9, 12, 8, 15, 13, 13, 11, 8, 7, 4, 8, 1, 3, 3, 5, 8],
    [11, 14, 6, 13, 10, 4, 2, 3, 5, 2, 10, 3, 6, 12, 14, 13],
    [0, 3, 15, 15, 9, 5, 2, 7, 0, 13, 1, 1, 7, 11, 14, 9],
    [0, 11, 0, 5, 3, 0, 10, 0, 0, 5, 0, 11, 13, 0, 6, 0],
    [6, 14, 10, 6, 11, 9, 2, 15, 10, 2, 6, 10, 5, 7, 14, 1],
    [10, 8, 11, 12, 4, 12, 0, 0, 6, 8, 5, 4, 12, 4, 0, 0],
    [6, 13, 7, 11, 6, 12, 5, 7, 10, 3, 9, 5, 10, 4, 11, 9],
    [4, 2, 2, 12, 7, 6, 12, 6, 12, 14, 14, 4, 9, 10, 4, 10],
    [3, 5, 8, 9, 9, 2, 7, 14, 13, 11, 8, 7, 7, 14, 9, 2],
    [2, 6, 14, 1, 1, 3, 14, 4, 14, 10, 2, 15, 15, 13, 2, 12],
    [0, 2, 8, 6, 8, 4, 10, 1, 0, 14, 8, 10, 8, 12, 6, 15],
    [11, 3, 5, 11, 4, 13, 5, 14, 5, 13, 11, 5, 12, 3, 11, 2],
    [4, 11, 4, 5, 3, 1, 9, 13, 12, 5, 12, 11, 13, 15, 7, 3],
    [1, 9, 12, 6, 10, 10, 8, 15, 15, 7, 4, 10, 6, 6, 8, 1],
], dtype=np.uint8)


def h_eb4(x1, x2, x3, x4, x5, x6):
    S = EB4_S
    x2 = (x2 + x1) & 15
    x4 = (x4 + x3) & 15
    x6 = (x6 + x5) & 15
    y1, y2, y3, y4, y5, y6 = S[0][x1], S[1][x2], S[2][x3], S[3][x4], S[4][x5], S[5][x6]
    z1 = S[6][(y6 + y1 + x3) & 15]
    z2 = S[7][(y5 + y2 + x4) & 15]
    z3 = S[8][(y2 + y3 + x5) & 15]
    z4 = S[9][(y1 + y4 + x6) & 15]
    z5 = S[10][(y4 + y5 + x1) & 15]
    z6 = S[11][(y3 + y6 + x2) & 15]
    t1 = (z1 + z2 + z3 + x6) & 15
    t2 = (z2 + z4 + x5) & 15
    t3 = (z3 + z4 + y1 + x4) & 15
    t4 = (z4 + z5 + z6 + x2) & 15
    t5 = (z5 + z1 + x1) & 15
    t6 = (z6 + z1 + y4 + x3) & 15
    return (S[12][t1] + S[13][t2] + S[14][t3] + S[15][t4] + S[16][t5] + S[17][t6]) & 15


CIPHERS = {"E4": (h_e4, 4), "Eb4": (h_eb4, 6)}


# ---------------------------------------------------------------- index helpers
def unpack(idx, n):
    """Index array -> list of n coordinate arrays (x1 most significant)."""
    idx = np.asarray(idx)
    return [((idx >> (4 * (n - 1 - k))) & 15).astype(np.uint8) for k in range(n)]


def shift_index(idx, n, j, s):
    """Index of x + s*e_j (mod 16 on coordinate j)."""
    sh = 4 * (n - 1 - j)
    xj = (idx >> sh) & 15
    return idx - (xj << sh) + (((xj + (s & 15)) & 15) << sh)


def index_dtype(n):
    return np.uint16 if n <= 4 else np.uint32


def build_table(h, n, delta, j, idx=None):
    """Dense T_{delta,j} over the inputs in idx (default: all 16^n inputs)."""
    if idx is None:
        idx = np.arange(16 ** n, dtype=np.uint32)
    x = unpack(idx, n)
    xd = list(x)
    xd[j] = (x[j] + delta) & 15
    return ((h(*x).astype(np.int16) - h(*xd)) & 15).astype(np.uint8)


def has_global_negation(h, n):
    """h(x + 8*(1,...,1)) == -h(x) for all x? (checked exhaustively)"""
    idx = np.arange(16 ** n, dtype=np.uint32)
    x = unpack(idx, n)
    y = [(v + 8) & 15 for v in x]
    return bool(np.all(h(*y) == ((-h(*x).astype(np.int16)) & 15)))


# ---------------------------------------------------------------- stores
class UniformStore:
    def __init__(self, cipher, deltas=range(1, 16)):
        h, self.n = CIPHERS[cipher]
        self.tables = {(d, j): build_table(h, self.n, d, j) for d in deltas for j in range(self.n)}

    def nbytes(self):
        return sum(t.nbytes for t in self.tables.values())

    def full(self, delta, j):
        return self.tables[(delta, j)]

    def lookup(self, delta, j, dh):
        return np.flatnonzero(self.full(delta, j) == dh)


def _invert(table, n):
    order = np.argsort(table, kind="stable").astype(index_dtype(n))
    offsets = np.searchsorted(table[order], np.arange(17)).astype(np.uint32)
    return order, offsets


class InvertedStore(UniformStore):
    def __init__(self, cipher, deltas=range(1, 16)):
        super().__init__(cipher, deltas)
        self.inv = {k: _invert(t, self.n) for k, t in self.tables.items()}
        self.tables = None  # only the bucketed form is kept

    def nbytes(self):
        return sum(o.nbytes + f.nbytes for o, f in self.inv.values())

    def lookup(self, delta, j, dh):
        order, offsets = self.inv[(delta, j)]
        return order[offsets[dh]:offsets[dh + 1]]


class SymmetricStore:
    """Stores T_{d,j} for d in 1..8 only; if h(x+8*1) == -h(x), only the half with x1 < 8.

    S1: T_{d,j}(x + 8*1) = -T_{d,j}(x)            (needs the global-negation property of h)
    S2: T_{16-d,j}(x)    = -T_{d,j}(x - d*e_j)    (holds for every h)
    """

    def __init__(self, cipher, use_half=None):
        h, self.n = CIPHERS[cipher]
        self.half = has_global_negation(h, self.n) if use_half is None else use_half
        size = 16 ** self.n
        idx = np.arange(size // 2 if self.half else size, dtype=np.uint32)
        self.tables = {(d, j): build_table(h, self.n, d, j, idx) for d in range(1, 9) for j in range(self.n)}

    def nbytes(self):
        return sum(t.nbytes for t in self.tables.values())

    def full(self, delta, j):
        n, size = self.n, 16 ** self.n
        if delta > 8:
            d = 16 - delta
            base = self.full(d, j)
            src = shift_index(np.arange(size, dtype=np.uint32), n, j, -d)
            return ((-base[src].astype(np.int16)) & 15).astype(np.uint8)
        t = self.tables[(delta, j)]
        if not self.half:
            return t
        # second half (x1 >= 8): T(x) = -T(x - 8*1), and x - 8*1 lies in the stored half
        hi = np.arange(size // 2, size, dtype=np.uint32)
        src = hi ^ np.uint32(int("8" * n, 16))  # subtracting 8 from every nibble == xor with 0x88..8
        return np.concatenate([t, ((-t[src].astype(np.int16)) & 15).astype(np.uint8)])

    def lookup(self, delta, j, dh):
        return np.flatnonzero(self.full(delta, j) == dh)


class SymmetricInvertedStore(SymmetricStore):
    """SymmetricStore plus an inverted index built lazily for each (delta, j) that is queried."""

    def __init__(self, cipher, use_half=None):
        super().__init__(cipher, use_half)
        self.cache = {}

    def nbytes(self):
        return super().nbytes() + sum(o.nbytes + f.nbytes for o, f in self.cache.values())

    def lookup(self, delta, j, dh):
        key = (delta, j)
        if key not in self.cache:
            self.cache[key] = _invert(self.full(delta, j), self.n)
        order, offsets = self.cache[key]
        return order[offsets[dh]:offsets[dh + 1]]


STORES = {
    "uniform": UniformStore,
    "inverted": InvertedStore,
    "symmetric": SymmetricStore,
    "symmetric_inverted": SymmetricInvertedStore,
}

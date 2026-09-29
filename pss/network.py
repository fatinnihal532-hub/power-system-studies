"""Network data model and admittance-matrix construction.

Data follow the MATPOWER case format (per unit on the system MVA base).
"""
from dataclasses import dataclass, field
import numpy as np

SLACK, PV, PQ = 3, 2, 1


@dataclass
class Network:
    base_mva: float
    bus: np.ndarray      # [id, type, Pd, Qd, Gs, Bs, Vm, Va(deg)]
    gen: np.ndarray      # [bus, Pg, Qg, Qmax, Qmin, Vg]
    branch: np.ndarray   # [from, to, r, x, b, tap]
    names: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.bus.shape[0]

    def idx(self, bus_id):
        """Zero-based index of a bus id."""
        return int(np.where(self.bus[:, 0] == bus_id)[0][0])

    def ybus(self):
        """Bus admittance matrix, with off-nominal taps and line charging (MATPOWER pi model)."""
        n = self.n
        Y = np.zeros((n, n), dtype=complex)
        for f, t, r, x, b, tap in self.branch:
            i, k = self.idx(f), self.idx(t)
            ys = 1.0 / complex(r, x)
            a = tap if tap != 0 else 1.0
            Y[i, i] += (ys + 1j * b / 2) / a**2
            Y[k, k] += ys + 1j * b / 2
            Y[i, k] -= ys / a
            Y[k, i] -= ys / a
        for row in self.bus:
            i = self.idx(row[0])
            Y[i, i] += complex(row[4], row[5]) / self.base_mva
        return Y


def load_ieee14():
    """IEEE 14-bus test system (MATPOWER case14, 100 MVA base)."""
    bus = np.array([
        # id type  Pd     Qd    Gs  Bs   Vm     Va
        [1, 3,   0.0,   0.0, 0, 0,  1.060,   0.00],
        [2, 2,  21.7,  12.7, 0, 0,  1.045,  -4.98],
        [3, 2,  94.2,  19.0, 0, 0,  1.010, -12.72],
        [4, 1,  47.8,  -3.9, 0, 0,  1.019, -10.33],
        [5, 1,   7.6,   1.6, 0, 0,  1.020,  -8.78],
        [6, 2,  11.2,   7.5, 0, 0,  1.070, -14.22],
        [7, 1,   0.0,   0.0, 0, 0,  1.062, -13.37],
        [8, 2,   0.0,   0.0, 0, 0,  1.090, -13.36],
        [9, 1,  29.5,  16.6, 0, 19, 1.056, -14.94],
        [10, 1,  9.0,   5.8, 0, 0,  1.051, -15.10],
        [11, 1,  3.5,   1.8, 0, 0,  1.057, -14.79],
        [12, 1,  6.1,   1.6, 0, 0,  1.055, -15.07],
        [13, 1, 13.5,   5.8, 0, 0,  1.050, -15.16],
        [14, 1, 14.9,   5.0, 0, 0,  1.036, -16.04],
    ], dtype=float)
    gen = np.array([
        # bus  Pg     Qg    Qmax  Qmin   Vg
        [1, 232.4, -16.9,  10,    0, 1.060],
        [2,  40.0,  42.4,  50,  -40, 1.045],
        [3,   0.0,  23.4,  40,    0, 1.010],
        [6,   0.0,  12.2,  24,   -6, 1.070],
        [8,   0.0,  17.4,  24,   -6, 1.090],
    ], dtype=float)
    branch = np.array([
        # from to   r        x        b      tap
        [1, 2, 0.01938, 0.05917, 0.0528, 0],
        [1, 5, 0.05403, 0.22304, 0.0492, 0],
        [2, 3, 0.04699, 0.19797, 0.0438, 0],
        [2, 4, 0.05811, 0.17632, 0.0340, 0],
        [2, 5, 0.05695, 0.17388, 0.0346, 0],
        [3, 4, 0.06701, 0.17103, 0.0128, 0],
        [4, 5, 0.01335, 0.04211, 0.0,    0],
        [4, 7, 0.0,     0.20912, 0.0,    0.978],
        [4, 9, 0.0,     0.55618, 0.0,    0.969],
        [5, 6, 0.0,     0.25202, 0.0,    0.932],
        [6, 11, 0.09498, 0.19890, 0.0,   0],
        [6, 12, 0.12291, 0.25581, 0.0,   0],
        [6, 13, 0.06615, 0.13027, 0.0,   0],
        [7, 8, 0.0,     0.17615, 0.0,    0],
        [7, 9, 0.0,     0.11001, 0.0,    0],
        [9, 10, 0.03181, 0.08450, 0.0,   0],
        [9, 14, 0.12711, 0.27038, 0.0,   0],
        [10, 11, 0.08205, 0.19207, 0.0,  0],
        [12, 13, 0.22092, 0.19988, 0.0,  0],
        [13, 14, 0.17093, 0.34802, 0.0,  0],
    ], dtype=float)
    return Network(100.0, bus, gen, branch)

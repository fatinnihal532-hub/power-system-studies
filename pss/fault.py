"""Short-circuit analysis with the bus impedance (Zbus) method.

Classical assumptions (stated so results can be reproduced):
  * sources are modelled as constant EMF behind subtransient reactance X'';
  * loads are converted to constant admittances at their pre-fault voltage;
  * pre-fault voltages come from the Newton-Raphson load flow (superposition method);
  * line charging and fixed shunts are kept in the network.
"""
from dataclasses import dataclass
import numpy as np

A = np.exp(2j * np.pi / 3)


@dataclass
class FaultResult:
    bus: int
    I_pu: complex            # fault current, pu (on base_mva / sqrt3 / kV)
    I_kA: float
    S_MVA: float             # fault level, MVA = sqrt3 * V_pre * I
    x_over_r: float          # Thevenin X/R at the fault point
    ip_kA: float             # IEC 60909 peak current, kappa * sqrt2 * I''k
    V_post: np.ndarray       # post-fault bus voltages, pu (positive sequence)


def base_current_kA(base_mva, kv):
    return base_mva / (np.sqrt(3) * kv)


def source_admittance(net, xdss, V_pre):
    """Positive-sequence Ybus augmented with generator X'' and load admittances."""
    Y = net.ybus().copy()
    for g in net.gen:
        i = net.idx(g[0])
        Y[i, i] += 1.0 / (1j * xdss[int(g[0])])
    for row in net.bus:
        i = net.idx(row[0])
        S = complex(row[2], row[3]) / net.base_mva
        if S != 0:
            Y[i, i] += np.conj(S) / abs(V_pre[i]) ** 2
    return Y


def kappa(x_over_r):
    """IEC 60909-0 peak factor, method (b) formula."""
    return 1.02 + 0.98 * np.exp(-3.0 / x_over_r)


def three_phase_faults(net, V_pre, xdss, kv, zf=0.0):
    """Bolted (or impedance zf) three-phase fault at every bus."""
    Y = source_admittance(net, xdss, V_pre)
    Z = np.linalg.inv(Y)
    out = []
    for i, row in enumerate(net.bus):
        bus = int(row[0])
        If = V_pre[i] / (Z[i, i] + zf)
        V_post = V_pre - Z[:, i] * If
        Ib = base_current_kA(net.base_mva, kv[bus])
        xr = Z[i, i].imag / Z[i, i].real
        I_kA = abs(If) * Ib
        out.append(FaultResult(bus, If, I_kA, abs(If) * abs(V_pre[i]) * net.base_mva, xr,
                               kappa(xr) * np.sqrt(2) * I_kA, V_post))
    return out


def fault_by_direct_solution(net, V_pre, xdss, bus, zf=1e-9):
    """Independent check: solve the nodal equations with a small fault impedance
    inserted and the machine EMFs as sources, instead of using Zbus."""
    Y = net.ybus()
    base = net.base_mva
    # machine internal EMFs from pre-fault operating point
    Yload = np.zeros(net.n, dtype=complex)
    for row in net.bus:
        S = complex(row[2], row[3]) / base
        Yload[net.idx(row[0])] = np.conj(S) / abs(V_pre[net.idx(row[0])]) ** 2
    Igen = Y @ V_pre + Yload * V_pre            # current each bus must receive from its machine
    Yf = Y + np.diag(Yload)
    Inj = np.zeros(net.n, dtype=complex)
    for g in net.gen:
        i = net.idx(g[0])
        ydd = 1.0 / (1j * xdss[int(g[0])])
        E = V_pre[i] + Igen[i] / ydd
        Yf[i, i] += ydd
        Inj[i] += E * ydd                        # Norton equivalent of the machine
    k = net.idx(bus)
    Yf[k, k] += 1.0 / zf
    V = np.linalg.solve(Yf, Inj)
    return V[k] / zf


def sequence_ybus(net, xdss, x0_gen, x0_factor=3.0):
    """Negative- and zero-sequence admittance matrices.

    Negative sequence: same network with X2 = X''.  Zero sequence (illustrative):
    lines have Z0 = x0_factor * Z1; transformers are taken as grounded-wye/grounded-wye
    (Z0 = Z1); machines are solidly grounded with reactance x0_gen.  Loads are left out
    of the zero-sequence network (ungrounded delta)."""
    n = net.n
    Y2 = net.ybus().copy()
    for g in net.gen:
        Y2[net.idx(g[0]), net.idx(g[0])] += 1.0 / (1j * xdss[int(g[0])])
    Y0 = np.zeros((n, n), dtype=complex)
    for f, t, r, x, b, tap in net.branch:
        i, k = net.idx(f), net.idx(t)
        m = 1.0 if tap != 0 or r == 0 else x0_factor
        y = 1.0 / (m * complex(r, x))
        Y0[i, i] += y + 1j * b / 2
        Y0[k, k] += y + 1j * b / 2
        Y0[i, k] -= y
        Y0[k, i] -= y
    for g in net.gen:
        Y0[net.idx(g[0]), net.idx(g[0])] += 1.0 / (1j * x0_gen)
    return Y2, Y0


def slg_fault(net, V_pre, xdss, x0_gen, bus, kv, zf=0.0):
    """Single-line-to-ground fault on phase a at `bus` (sequence networks in series)."""
    Y1 = source_admittance(net, xdss, V_pre)
    Y2, Y0 = sequence_ybus(net, xdss, x0_gen)
    k = net.idx(bus)
    Z1 = np.linalg.inv(Y1)[k, k]
    Z2 = np.linalg.inv(Y2)[k, k]
    Z0 = np.linalg.inv(Y0)[k, k]
    Ia0 = V_pre[k] / (Z1 + Z2 + Z0 + 3 * zf)
    Iabc = np.array([[1, 1, 1], [1, A**2, A], [1, A, A**2]]) @ np.array([Ia0, Ia0, Ia0])
    return {
        "bus": bus,
        "Z1": Z1, "Z2": Z2, "Z0": Z0,
        "Ia_pu": Iabc[0],
        "Ia_kA": abs(Iabc[0]) * base_current_kA(net.base_mva, kv[bus]),
        "Ib_pu": Iabc[1], "Ic_pu": Iabc[2],
    }

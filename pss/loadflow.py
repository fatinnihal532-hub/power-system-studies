"""Full Newton-Raphson load flow in polar coordinates."""
from dataclasses import dataclass
import numpy as np
from .network import SLACK, PV, PQ


@dataclass
class LoadFlowResult:
    V: np.ndarray            # complex bus voltages, pu
    iterations: int
    mismatch_history: list
    S_gen: np.ndarray        # complex generation per bus, MVA
    S_load: np.ndarray       # complex load per bus, MVA
    flows: list              # (from, to, S_from MVA, S_to MVA)
    losses: complex          # total branch losses, MVA

    @property
    def vm(self):
        return np.abs(self.V)

    @property
    def va_deg(self):
        return np.degrees(np.angle(self.V))


def _power(V, Y):
    return V * np.conj(Y @ V)


def newton_raphson(net, tol=1e-8, max_iter=20):
    """Solve the load flow. Generator reactive limits are not enforced, matching the
    published IEEE 14-bus solution (MATPOWER runpf default)."""
    n, base = net.n, net.base_mva
    Y = net.ybus()
    types = net.bus[:, 1].astype(int)

    # scheduled injections
    Psch = -net.bus[:, 2] / base
    Qsch = -net.bus[:, 3] / base
    vm = np.ones(n)
    for g in net.gen:
        i = net.idx(g[0])
        Psch[i] += g[1] / base
        vm[i] = g[5]
    va = np.zeros(n)

    pv_pq = np.where(types != SLACK)[0]
    pq = np.where(types == PQ)[0]
    history = []

    for it in range(1, max_iter + 1):
        V = vm * np.exp(1j * va)
        S = _power(V, Y)
        dP = Psch[pv_pq] - S.real[pv_pq]
        dQ = Qsch[pq] - S.imag[pq]
        mis = np.concatenate([dP, dQ])
        history.append(float(np.max(np.abs(mis))))
        if history[-1] < tol:
            break

        # Jacobian via complex derivatives (dS/dVa, dS/dVm)
        Ibus = Y @ V
        diagV = np.diag(V)
        diagI = np.diag(Ibus)
        diagVn = np.diag(V / np.abs(V))
        dS_dVa = 1j * diagV @ np.conj(diagI - Y @ diagV)
        dS_dVm = diagV @ np.conj(Y @ diagVn) + np.conj(diagI) @ diagVn
        J = np.block([
            [dS_dVa.real[np.ix_(pv_pq, pv_pq)], dS_dVm.real[np.ix_(pv_pq, pq)]],
            [dS_dVa.imag[np.ix_(pq, pv_pq)],    dS_dVm.imag[np.ix_(pq, pq)]],
        ])
        dx = np.linalg.solve(J, mis)
        va[pv_pq] += dx[:len(pv_pq)]
        vm[pq] += dx[len(pv_pq):]
    else:
        raise RuntimeError("Newton-Raphson did not converge")

    V = vm * np.exp(1j * va)
    S = _power(V, Y) * base
    S_load = (net.bus[:, 2] + 1j * net.bus[:, 3]).astype(complex)
    S_gen = S + S_load          # fixed shunts are part of Ybus, so they appear in S

    flows, losses = [], 0j
    for f, t, r, x, b, tap in net.branch:
        i, k = net.idx(f), net.idx(t)
        a = tap if tap != 0 else 1.0
        ys = 1.0 / complex(r, x)
        If = ((ys + 1j * b / 2) / a**2) * V[i] - (ys / a) * V[k]
        It = (ys + 1j * b / 2) * V[k] - (ys / a) * V[i]
        Sf, St = V[i] * np.conj(If) * base, V[k] * np.conj(It) * base
        flows.append((int(f), int(t), Sf, St))
        losses += Sf + St
    return LoadFlowResult(V, it, history, S_gen, S_load, flows, losses)

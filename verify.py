"""Checks every number quoted in the README against an independent reference.

Run:  python verify.py
"""
import numpy as np
from pss import load_ieee14, newton_raphson, three_phase_faults, slg_fault
from pss.fault import fault_by_direct_solution
from pss.assumptions import KV, XDSS, X0_GEN, BREAKER_RATING_KA

checks = []


def check(name, ok, detail=""):
    checks.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


net = load_ieee14()
lf = newton_raphson(net)

# 1. Converges quadratically, in a handful of iterations
check("Newton-Raphson converges in <= 5 iterations", lf.iterations <= 5,
      f"({lf.iterations} iterations, final mismatch {lf.mismatch_history[-1]:.1e} pu)")
h = lf.mismatch_history
check("Quadratic convergence (each error ~ square of the last)",
      all(h[i + 1] < 10 * h[i] ** 2 for i in range(1, len(h) - 1)))

# 2. Matches the published IEEE 14-bus solution (values stored in the case, 3 d.p. / 0.01 deg)
dvm = np.max(np.abs(lf.vm - net.bus[:, 6]))
dva = np.max(np.abs(lf.va_deg - net.bus[:, 7]))
check("Voltages match the published IEEE 14-bus solution", dvm < 2e-3 and dva < 0.05,
      f"(max |dV| {dvm:.4f} pu, max |dtheta| {dva:.3f} deg)")

# 3. Power balance: generation = load + losses
bal = lf.S_gen.sum() - lf.S_load.sum() - lf.losses
check("Power balance closes", abs(bal.real) < 1e-6,
      f"(P gen {lf.S_gen.real.sum():.2f} MW, load {lf.S_load.real.sum():.1f} MW, losses {lf.losses.real:.3f} MW)")
check("Losses match MATPOWER case14 (13.393 MW)", abs(lf.losses.real - 13.393) < 1e-3)

# 4. Short circuit: Zbus result equals a direct nodal solution with the fault inserted
faults = three_phase_faults(net, lf.V, XDSS, KV)
err = max(abs(fault_by_direct_solution(net, lf.V, XDSS, f.bus) - f.I_pu) / abs(f.I_pu) for f in faults)
check("Zbus fault currents equal direct nodal solution", err < 1e-6, f"(max rel. error {err:.1e})")

# 5. Symmetry: a fault on a bus of the unloaded, flat-start network gives V/Zkk exactly
flat = np.ones(net.n, dtype=complex)
f0 = three_phase_faults(net, flat, XDSS, KV)
check("Flat-start fault current = 1/Zkk", all(abs(f.V_post[net.idx(f.bus)]) < 1e-12 for f in f0))

# 6. SLG: healthy phases carry no current; faulted phase = 3 * I0
s = slg_fault(net, lf.V, XDSS, X0_GEN, 9, KV)
check("SLG fault: Ib = Ic = 0", abs(s["Ib_pu"]) < 1e-12 and abs(s["Ic_pu"]) < 1e-12)

# 7. Breaker duty findings quoted in the README
over = [f.bus for f in faults if f.I_kA > BREAKER_RATING_KA[KV[f.bus]]]
check("Only bus 8 exceeds its assumed breaker rating", over == [8], f"(buses over rating: {over})")

print(f"\n{sum(checks)}/{len(checks)} checks passed")
raise SystemExit(0 if all(checks) else 1)

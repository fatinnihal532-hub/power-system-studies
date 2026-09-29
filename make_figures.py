"""Regenerate the figures in docs/ and the result tables in results/."""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pss import load_ieee14, newton_raphson, three_phase_faults, slg_fault
from pss.assumptions import KV, XDSS, X0_GEN, BREAKER_RATING_KA

plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "fixed", "font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
NAVY, GOLD, RED, GREY = "#1F3864", "#C9971C", "#B3261E", "#777777"


def compact_svg(path):
    """Shrink a matplotlib SVG: round coordinates to 0.1 pt and drop metadata."""
    s = open(path).read()
    s = re.sub(r"<metadata>.*?</metadata>\s*", "", s, flags=re.S)
    s = re.sub(r"-?\d+\.\d{2,}", lambda m: f"{float(m.group()):.1f}".rstrip("0").rstrip("."), s)
    s = re.sub(r"\n\s+", "\n", s)
    open(path, "w").write(s)


os.makedirs("docs", exist_ok=True)
os.makedirs("results", exist_ok=True)

net = load_ieee14()
lf = newton_raphson(net)
buses = net.bus[:, 0].astype(int)

# 1. voltage profile
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(buses, lf.vm, "o-", color=NAVY, label="This solver (Newton-Raphson)")
ax.plot(buses, net.bus[:, 6], "x", color=GOLD, ms=8, mew=2, label="Published IEEE solution")
ax.set_xticks(buses)
ax.set_xlabel("Bus")
ax.set_ylabel("Voltage (pu)")
ax.set_ylim(0.99, 1.10)
ax.set_title("IEEE 14-bus voltage profile", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("docs/voltage_profile.svg")
compact_svg("docs/voltage_profile.svg")
plt.close(fig)

# 2. convergence
fig, ax = plt.subplots(figsize=(4.2, 3.0))
it = np.arange(len(lf.mismatch_history))
ax.semilogy(it, lf.mismatch_history, "o-", color=NAVY)
ax.set_xticks(it)
ax.set_xlabel("Iteration")
ax.set_ylabel("Max power mismatch (pu)")
ax.set_title("Quadratic convergence", loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig("docs/convergence.svg")
compact_svg("docs/convergence.svg")
plt.close(fig)

# 3. fault levels vs switchgear rating
faults = three_phase_faults(net, lf.V, XDSS, KV)
I = np.array([f.I_kA for f in faults])
rating = np.array([BREAKER_RATING_KA[KV[b]] for b in buses])
colors = [RED if i > r else NAVY for i, r in zip(I, rating)]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(buses, I, color=colors, width=0.65)
for b, r in zip(buses, rating):
    ax.plot([b - 0.4, b + 0.4], [r, r], color=GOLD, lw=2)
ax.plot([], [], color=GOLD, lw=2, label="Breaker rating (assumed)")
from matplotlib.patches import Patch
handles, labels = ax.get_legend_handles_labels()
handles += [Patch(color=NAVY, label="Within rating"), Patch(color=RED, label="Exceeds rating")]
ax.set_xticks(buses)
ax.set_xticklabels([f"{b}\n{KV[b]} kV" for b in buses], fontsize=7.5)
ax.set_ylabel("Three-phase fault current (kA)")
ax.set_title("Symmetrical fault current I''k vs switchgear rating", loc="left", fontweight="bold")
ax.legend(handles=handles, frameon=False)
fig.tight_layout()
fig.savefig("docs/fault_levels.svg")
compact_svg("docs/fault_levels.svg")
plt.close(fig)

# tables
with open("results/load_flow.csv", "w") as fh:
    fh.write("bus,vm_pu,va_deg,p_gen_mw,q_gen_mvar,p_load_mw,q_load_mvar\n")
    for i, b in enumerate(buses):
        g, l = lf.S_gen[i] + 0j, lf.S_load[i]
        g = complex(round(g.real, 2) + 0.0, round(g.imag, 2) + 0.0)   # avoid printing -0.00
        fh.write(f"{b},{lf.vm[i]:.4f},{lf.va_deg[i]:.3f},{g.real:.2f},{g.imag:.2f},{l.real:.2f},{l.imag:.2f}\n")
with open("results/line_flows.csv", "w") as fh:
    fh.write("from,to,p_from_mw,q_from_mvar,p_loss_mw,q_loss_mvar\n")
    for f, t, Sf, St in lf.flows:
        fh.write(f"{f},{t},{Sf.real:.2f},{Sf.imag:.2f},{(Sf+St).real:.3f},{(Sf+St).imag:.3f}\n")
with open("results/fault_study.csv", "w") as fh:
    fh.write("bus,kv,fault_mva,ik_ka,x_over_r,ip_ka,breaker_ka,slg_ka\n")
    for f in faults:
        s = slg_fault(net, lf.V, XDSS, X0_GEN, f.bus, KV)
        fh.write(f"{f.bus},{KV[f.bus]},{f.S_MVA:.1f},{f.I_kA:.2f},{f.x_over_r:.2f},{f.ip_kA:.2f},"
                 f"{BREAKER_RATING_KA[KV[f.bus]]},{s['Ia_kA']:.2f}\n")
print("figures in docs/, tables in results/")

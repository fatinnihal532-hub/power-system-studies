import numpy as np
import pytest
from pss import Network, load_ieee14, newton_raphson, three_phase_faults, slg_fault
from pss.fault import fault_by_direct_solution
from pss.assumptions import KV, XDSS, X0_GEN


@pytest.fixture(scope="module")
def lf():
    net = load_ieee14()
    return net, newton_raphson(net)


def two_bus():
    """Slack bus feeding a PQ load through one line: solvable by hand."""
    bus = np.array([[1, 3, 0, 0, 0, 0, 1.0, 0], [2, 1, 50, 20, 0, 0, 1.0, 0]], float)
    gen = np.array([[1, 0, 0, 99, -99, 1.0]], float)
    branch = np.array([[1, 2, 0.02, 0.08, 0.0, 0]], float)
    return Network(100.0, bus, gen, branch)


def test_two_bus_by_hand():
    net = two_bus()
    r = newton_raphson(net)
    V2 = r.V[1]
    # KVL check: V1 - V2 = Z * I, with I = conj(S_load / V2)
    I = np.conj((0.5 + 0.2j) / V2)
    assert abs(1.0 - V2 - complex(0.02, 0.08) * I) < 1e-10


def test_ieee14_matches_published(lf):
    net, r = lf
    assert np.max(np.abs(r.vm - net.bus[:, 6])) < 2e-3
    assert np.max(np.abs(r.va_deg - net.bus[:, 7])) < 0.05


def test_losses_and_balance(lf):
    net, r = lf
    assert abs(r.losses.real - 13.393) < 1e-3
    assert abs((r.S_gen.sum() - r.S_load.sum() - r.losses).real) < 1e-6


def test_matches_pandapower(lf):
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    net_pp = pn.case14()
    pp.runpp(net_pp, enforce_q_lims=False, tolerance_mva=1e-10)
    _, r = lf
    assert np.allclose(net_pp.res_bus.vm_pu.values, r.vm, atol=1e-8)
    assert np.allclose(net_pp.res_bus.va_degree.values, r.va_deg, atol=1e-6)


def test_zbus_equals_direct_solution(lf):
    net, r = lf
    for f in three_phase_faults(net, r.V, XDSS, KV):
        d = fault_by_direct_solution(net, r.V, XDSS, f.bus)
        assert abs(d - f.I_pu) / abs(f.I_pu) < 1e-6


def test_two_bus_fault_by_hand():
    net = two_bus()
    V = np.array([1.0, 1.0], complex)
    # unloaded pre-fault: source X''=0.1 behind bus 1; fault at bus 2
    net.bus[1, 2:4] = 0
    f = three_phase_faults(net, V, {1: 0.1}, {1: 132, 2: 132})[1]
    assert abs(f.I_pu - 1.0 / complex(0.02, 0.18)) < 1e-12


def test_slg_sequence_relations(lf):
    net, r = lf
    s = slg_fault(net, r.V, XDSS, X0_GEN, 4, KV)
    assert abs(s["Ib_pu"]) < 1e-12 and abs(s["Ic_pu"]) < 1e-12
    assert abs(s["Ia_pu"] - 3 * r.V[net.idx(4)] / (s["Z1"] + s["Z2"] + s["Z0"])) < 1e-12

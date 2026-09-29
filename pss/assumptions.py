"""Study assumptions that are not part of the IEEE 14-bus load-flow data.

The IEEE 14-bus case carries no machine impedances or voltage levels, so these are
stated explicitly. Voltage levels follow a 132/33/11 kV structure, typical of the
Bangladesh transmission and sub-transmission network.
"""
# nominal voltage of each bus, kV
KV = {1: 132, 2: 132, 3: 132, 4: 132, 5: 132,
      6: 33, 7: 33, 8: 11, 9: 33, 10: 33, 11: 33, 12: 33, 13: 33, 14: 33}

# subtransient reactance X'' of each machine, pu on the 100 MVA system base
XDSS = {1: 0.20, 2: 0.20, 3: 0.25, 6: 0.25, 8: 0.25}

# machine zero-sequence reactance, pu (solidly grounded neutrals)
X0_GEN = 0.08

# switchgear rated short-circuit breaking current, kA (common IEC ratings)
BREAKER_RATING_KA = {132: 31.5, 33: 25.0, 11: 25.0}

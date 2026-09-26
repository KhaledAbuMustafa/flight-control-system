"""
Flying-qualities criteria for the five eigenmodes (MIL-F-8785C, Class I = light aircraft).

Flight phase categories:
    B = cruise, climb, descent (gradual manoeuvres)
    C = take-off, approach, landing (precise flight path control)

Levels:
    1 = clearly adequate          (Cooper-Harper ~1-3.5)
    2 = adequate, higher workload (Cooper-Harper ~3.5-6.5)
    3 = controllable              (Cooper-Harper ~6.5-9)
    4 = worse than Level 3 (not acceptable)

Source: MIL-F-8785C (5 Nov 1980), all values checked against the norm text:
    phugoid        3.2.1.2               (p. 12)
    short period   3.2.2.1.2, Table IV   (p. 13)
    dutch roll     3.3.1.1,   Table VI   (p. 22)
    roll mode      3.3.1.2,   Table VII  (p. 23)
    spiral         3.3.1.3,   Table VIII (p. 23)

Not implemented (simplifications):
    - Table VI: extra zeta*wn requirement when wn^2*|phi/beta| > 20 (rad/s)^2
    - 3.3.1.4 coupled roll-spiral oscillation (not present in our model)
    - short-period frequency / CAP criteria (3.2.2.1.1, figures 1-3)
"""
import numpy as np

# ------------------------------------------------------------------
# Limits per category. Each level lists the conditions that must hold.
# ------------------------------------------------------------------
SHORT_PERIOD_ZETA = {        # (min, max) damping ratio
    "B": {1: (0.30, 2.00), 2: (0.20, 2.00), 3: (0.15, np.inf)},
    "C": {1: (0.35, 1.30), 2: (0.25, 2.00), 3: (0.15, np.inf)},
}
PHUGOID = {                  # all categories
    1: {"zeta_min": 0.04},
    2: {"zeta_min": 0.0},
    3: {"t_double_min": 55.0},
}
DUTCH_ROLL = {               # (zeta_min, zeta*wn_min, wn_min), Class I
    "B": {1: (0.08, 0.15, 0.4), 2: (0.02, 0.05, 0.4), 3: (0.0, -np.inf, 0.4)},
    "C": {1: (0.08, 0.15, 1.0), 2: (0.02, 0.05, 0.4), 3: (0.0, -np.inf, 0.4)},
}
ROLL_TAU_MAX = {             # maximum roll-mode time constant [s], Class I
    "B": {1: 1.4, 2: 3.0, 3: 10.0},
    "C": {1: 1.0, 2: 1.4, 3: 10.0},
}
SPIRAL_T2_MIN = {            # minimum time to double amplitude [s] (if unstable)
    "B": {1: 20.0, 2: 12.0, 3: 4.0},
    "C": {1: 12.0, 2: 8.0, 3: 4.0},
}


def _is_oscillatory(m):
    return "period" in m


def _t_double(m):
    """Time to double amplitude (inf if the mode is stable)."""
    s = m["eigenvalue"].real
    return np.log(2) / s if s > 0 else np.inf


def level_short_period(m, cat):
    if not _is_oscillatory(m):                    # aperiodic
        return 4 if m["eigenvalue"].real > 0 else 3
    for lvl in (1, 2, 3):
        lo, hi = SHORT_PERIOD_ZETA[cat][lvl]
        if lo <= m["zeta"] <= hi:
            return lvl
    return 4


def level_phugoid(m, cat):
    zeta = m["zeta"] if _is_oscillatory(m) else (1.0 if m["eigenvalue"].real < 0 else -1.0)
    if zeta >= PHUGOID[1]["zeta_min"]:
        return 1
    if zeta >= PHUGOID[2]["zeta_min"]:
        return 2
    if _t_double(m) >= PHUGOID[3]["t_double_min"]:
        return 3
    return 4


def level_dutch_roll(m, cat):
    if not _is_oscillatory(m):
        return 4
    for lvl in (1, 2, 3):
        z_min, zw_min, wn_min = DUTCH_ROLL[cat][lvl]
        if m["zeta"] >= z_min and m["zeta"] * m["wn"] >= zw_min and m["wn"] >= wn_min:
            return lvl
    return 4


def level_roll(m, cat):
    if m["eigenvalue"].real >= 0:
        return 4
    for lvl in (1, 2, 3):
        if m["tau"] <= ROLL_TAU_MAX[cat][lvl]:
            return lvl
    return 4


def level_spiral(m, cat):
    t2 = _t_double(m)                             # inf when stable
    for lvl in (1, 2, 3):
        if t2 >= SPIRAL_T2_MIN[cat][lvl]:
            return lvl
    return 4


CHECKS = {
    "short_period": level_short_period,
    "phugoid": level_phugoid,
    "roll": level_roll,
    "dutch_roll": level_dutch_roll,
    "spiral": level_spiral,
}


def evaluate(modes, category="B"):
    """Return {mode: level} and the overall (worst) level."""
    levels = {name: (CHECKS[name](modes[name], category) if name in modes else 4)
              for name in CHECKS}
    return levels, max(levels.values())


if __name__ == "__main__":
    from aircraft import p
    from trim_6dof import trim_6dof, linearize_6dof
    from modes import identify_modes

    names = ["short_period", "phugoid", "roll", "dutch_roll", "spiral"]
    header = "  V [m/s] | cat | " + " | ".join(f"{n:12s}" for n in names) + " | overall"
    print("C172, reference CG (Level 1 = good, 2 = acceptable, 3 = controllable, 4 = not acceptable)")
    print(header)
    for V in [30.0, 40.0, 50.0, 60.0]:
        modes = identify_modes(linearize_6dof(*trim_6dof(V))[0], V)
        for cat in (["B", "C"] if V <= 35 else ["B"]):
            levels, overall = evaluate(modes, cat)
            print(f"  {V:7.0f} |  {cat}  | " + " | ".join(f"Level {levels[n]:<6d}" for n in names)
                  + f" | Level {overall}")
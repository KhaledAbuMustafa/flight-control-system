"""
Automatic identification of the five classic eigenmodes from a 12x12 A matrix.

Idea:
  1) Keep only the 8 "dynamic" states (u, w, q, theta | v, p, r, phi).
     x_N, y_E, psi, h do not feed back (constant air density) -> eigenvalues 0.
  2) For every eigenvalue look at its eigenvector: does it mostly move
     longitudinal states or lateral states?  -> longitudinal / lateral mode
  3) Inside each group, sort by speed and shape:
       longitudinal: fast oscillation = short period, slow oscillation = phugoid
       lateral:      fast real = roll, slow real = spiral, oscillation = dutch roll
"""
import numpy as np
import flightdynamics_6dof as m6

LON = [m6.U, m6.W, m6.Q, m6.THETA]
LAT = [m6.V_, m6.P, m6.R, m6.PHI]
DYN = LON + LAT                                      # 8 dynamic states


def _mode_info(lam):
    """Characteristic numbers of one eigenvalue."""
    wn = abs(lam)
    info = {"eigenvalue": lam, "wn": wn}
    if abs(lam.imag) > 1e-6:                          # oscillation
        info["zeta"] = -lam.real / wn
        info["period"] = 2 * np.pi / abs(lam.imag)
    else:                                            # pure exponential
        info["zeta"] = 1.0 if lam.real < 0 else -1.0
        info["tau"] = -1.0 / lam.real if lam.real != 0 else np.inf       # time constant
        info["t_double"] = np.log(2) / lam.real if lam.real > 0 else np.inf  # divergence
    return info


def identify_modes(A, V):
    """
    Returns a dict  {mode name: info}  with the keys
    'short_period', 'phugoid', 'roll', 'dutch_roll', 'spiral' (when found).
    """
    A8 = A[np.ix_(DYN, DYN)]
    lam, vec = np.linalg.eig(A8)

    # Make velocities comparable to angles: divide u, w, v by V (-> dimensionless)
    scale = np.ones(8)
    scale[[0, 1, 4]] = 1.0 / V
    vec = vec * scale[:, None]

    lon, lat = [], []
    for k in range(len(lam)):
        if lam[k].imag < -1e-6:                       # each complex pair only once
            continue
        share_lon = np.linalg.norm(vec[:4, k]) / np.linalg.norm(vec[:, k])
        (lon if share_lon > 0.5 else lat).append(lam[k])

    modes = {}

    # ---- longitudinal ----
    osc = sorted([l for l in lon if abs(l.imag) > 1e-6], key=abs)
    real = sorted([l for l in lon if abs(l.imag) <= 1e-6], key=abs)
    if len(osc) == 2:                                 # normal case
        modes["phugoid"], modes["short_period"] = _mode_info(osc[0]), _mode_info(osc[1])
    elif len(osc) == 1:                               # one pair turned into two real poles
        if abs(osc[0]) < 1.0:                         # the remaining pair is slow -> phugoid
            modes["phugoid"] = _mode_info(osc[0])
            modes["short_period"] = _mode_info(max(real, key=lambda l: l.real))  # the critical one
        else:
            modes["short_period"] = _mode_info(osc[0])
            modes["phugoid"] = _mode_info(max(real, key=lambda l: l.real))
    elif len(osc) == 0 and real:                      # all real (e.g. CG behind neutral point)
        critical = max(real, key=lambda l: l.real)                    # pitch divergence
        rest = [l for l in real if l is not critical]
        modes["short_period"] = _mode_info(critical)
        if rest:
            modes["phugoid"] = _mode_info(min(rest, key=abs))           # slowest remaining root

    # ---- lateral ----
    osc = [l for l in lat if abs(l.imag) > 1e-6]
    real = sorted([l for l in lat if abs(l.imag) <= 1e-6], key=abs)
    if osc:
        modes["dutch_roll"] = _mode_info(osc[0])
    if len(real) >= 2:
        modes["spiral"], modes["roll"] = _mode_info(real[0]), _mode_info(real[-1])

    return modes


def print_modes(modes):
    names = {"short_period": "Short period", "phugoid": "Phugoid",
             "roll": "Roll", "dutch_roll": "Dutch roll", "spiral": "Spiral"}
    for key, label in names.items():
        if key not in modes:
            print(f"  {label:13s}: not found")
            continue
        m = modes[key]
        if "period" in m:
            print(f"  {label:13s}: wn = {m['wn']:6.3f} rad/s, zeta = {m['zeta']:6.3f}, period = {m['period']:5.1f} s")
        elif m["eigenvalue"].real < 0:
            print(f"  {label:13s}: time constant = {m['tau']:6.2f} s  (stable)")
        else:
            print(f"  {label:13s}: time to double = {m['t_double']:6.2f} s  (UNSTABLE)")


if __name__ == "__main__":
    import c172_params as p
    from trim_6dof import trim_6dof, linearize_6dof

    def analyse(title, V, psi_dot=0.0, dh=0.0):
        p.dh_cg = dh
        try:
            A, _ = linearize_6dof(*trim_6dof(V, psi_dot))
        finally:
            p.dh_cg = 0.0
        print(title)
        print_modes(identify_modes(A, V))
        print()

    analyse("Reference: V = 50 m/s, straight, reference CG", 50.0)
    analyse("Aft CG (dh = +0.16): close to the neutral point", 50.0, dh=0.16)
    analyse("Behind neutral point (dh = +0.20): unstable", 50.0, dh=0.20)
    analyse("In a 3°/s turn (longitudinal and lateral coupled)", 50.0, psi_dot=np.deg2rad(3.0))

    print("Speed sweep (reference CG):")
    print("   V  | short period wn/zeta | phugoid wn/zeta | roll tau | dutch roll wn/zeta | spiral")
    for V in [30.0, 40.0, 50.0, 60.0]:
        A, _ = linearize_6dof(*trim_6dof(V))
        m = identify_modes(A, V)
        sp, ph, ro, dr, sl = m["short_period"], m["phugoid"], m["roll"], m["dutch_roll"], m["spiral"]
        print(f"  {V:3.0f}  |   {sp['wn']:5.2f} / {sp['zeta']:4.2f}      |  {ph['wn']:5.3f} / {ph['zeta']:5.3f} |"
              f"  {ro['tau']:5.3f} s |    {dr['wn']:5.2f} / {dr['zeta']:4.2f}     | "
              + (f"tau = {sl['tau']:5.1f} s (stable)" if sl["eigenvalue"].real < 0
                 else f"doubles in {sl['t_double']:4.1f} s (unstable)"))
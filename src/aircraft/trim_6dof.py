"""
Trim and linearization for the 6-DOF model.

Steady, level, coordinated flight at airspeed V with turn rate psi_dot:
    psi_dot = 0  -> straight and level flight
    psi_dot != 0 -> steady coordinated turn (no sideslip, no height change)

Unknowns:   alpha, phi, delta_e, delta_a, delta_r, T        (6)
Conditions: u_dot = v_dot = w_dot = p_dot = q_dot = r_dot = 0   (6)
Fixed:      beta = 0 (coordinated), gamma = 0 (level)
"""
import numpy as np
from scipy.optimize import fsolve
import flightdynamics_6dof as m6


def build_state(V, alpha, phi, psi_dot, h=1000.0, beta=0.0):
    """Construct the 12-state vector for a steady, level turn."""
    # Body velocities from V, alpha, beta
    u = V * np.cos(alpha) * np.cos(beta)
    v = V * np.sin(beta)
    w = V * np.sin(alpha) * np.cos(beta)

    # Pitch angle for gamma = 0 (flight path horizontal, also in a banked turn)
    a = np.cos(alpha) * np.cos(beta)
    b = np.sin(phi) * np.sin(beta) + np.cos(phi) * np.sin(alpha) * np.cos(beta)
    theta = np.arctan2(b, a)

    # Body rates of a steady turn: the aircraft turns about the vertical axis with psi_dot
    p = -psi_dot * np.sin(theta)
    q = psi_dot * np.sin(phi) * np.cos(theta)
    r = psi_dot * np.cos(phi) * np.cos(theta)

    return np.array([u, v, w, p, q, r, phi, theta, 0.0, 0.0, 0.0, h])


def trim_residuals(z, V, psi_dot, h):
    alpha, phi, de, da, dr, T = z
    x = build_state(V, alpha, phi, psi_dot, h)
    c = np.array([de, da, dr, T])
    return m6.f(x, c)[:6]                            # u, v, w, p, q, r derivatives


def trim_6dof(V, psi_dot=0.0, h=1000.0):
    """
    Returns:
        x0 : 12-state trim vector
        c0 : trim inputs [delta_e, delta_a, delta_r, T]
    """
    phi_guess = np.arctan(V * psi_dot / m6.p.g)      # textbook bank angle for this turn rate
    z0 = [np.deg2rad(2.0), phi_guess, 0.0, 0.0, 0.0, 1000.0]
    z, info, ok, msg = fsolve(trim_residuals, z0, args=(V, psi_dot, h), full_output=True)
    if ok != 1:
        raise RuntimeError(f"6-DOF trim failed (V = {V}, psi_dot = {psi_dot}): {msg}")
    alpha, phi, de, da, dr, T = z
    return build_state(V, alpha, phi, psi_dot, h), np.array([de, da, dr, T])


def linearize_6dof(x0, c0):
    """Central-difference Jacobians A (12x12) and B (12x4)."""
    n, m = len(x0), len(c0)
    A, B = np.zeros((n, n)), np.zeros((n, m))
    for i in range(n):
        d = np.zeros(n); d[i] = 1e-6 * max(1.0, abs(x0[i]))
        A[:, i] = (m6.f(x0 + d, c0) - m6.f(x0 - d, c0)) / (2 * d[i])
    for j in range(m):
        d = np.zeros(m); d[j] = 1e-6 * max(1.0, abs(c0[j]))
        B[:, j] = (m6.f(x0, c0 + d) - m6.f(x0, c0 - d)) / (2 * d[j])
    return A, B


def describe(name, x0, c0):
    V, alpha, beta = m6.air_data(x0)
    xdot = m6.f(x0, c0)
    print(f"{name}")
    print(f"  alpha = {np.rad2deg(alpha):6.2f}°   theta = {np.rad2deg(x0[m6.THETA]):6.2f}°   "
          f"phi = {np.rad2deg(x0[m6.PHI]):6.2f}°   beta = {np.rad2deg(beta):5.2f}°")
    print(f"  delta_e = {np.rad2deg(c0[0]):6.2f}°   delta_a = {np.rad2deg(c0[1]):6.2f}°   "
          f"delta_r = {np.rad2deg(c0[2]):6.2f}°   T = {c0[3]:7.1f} N")
    print(f"  load factor n = {1/np.cos(x0[m6.PHI]):.3f}   psi_dot = {np.rad2deg(xdot[m6.PSI]):.2f}°/s   "
          f"h_dot = {xdot[m6.H]:.1e} m/s   max residual = {np.abs(xdot[:8]).max():.1e}")


if __name__ == "__main__":
    from trim import trim

    V = 50.0

    # ---------- straight and level: compare with the 3-DOF trim ----------
    x0, c0 = trim_6dof(V)
    describe(f"Straight and level, V = {V} m/s (6-DOF)", x0, c0)
    x3, u3 = trim(V)
    print(f"  3-DOF reference: alpha = {np.rad2deg(x3[2]):6.2f}°   delta_e = {np.rad2deg(u3[0]):6.2f}°   T = {u3[1]:7.1f} N\n")

    # ---------- standard rate turn: 3 deg/s to the right ----------
    psi_dot = np.deg2rad(3.0)
    xt, ct = trim_6dof(V, psi_dot)
    describe(f"Coordinated right turn, 3°/s, V = {V} m/s", xt, ct)
    print(f"  turn radius = V/psi_dot = {V/psi_dot:.0f} m, full circle in {360/3:.0f} s")

    # ---------- coupling: in the turn, longitudinal and lateral blocks couple ----------
    LON = [m6.U, m6.W, m6.Q, m6.THETA]
    LAT = [m6.V_, m6.P, m6.R, m6.PHI]
    for label, (xx, cc) in [("straight", (x0, c0)), ("turn", (xt, ct))]:
        A, _ = linearize_6dof(xx, cc)
        print(f"\nCoupling in A ({label}): max |lat -> lon| = {np.abs(A[np.ix_(LON, LAT)]).max():.3f}, "
              f"max |lon -> lat| = {np.abs(A[np.ix_(LAT, LON)]).max():.3f}")

    # ---------- open-loop check: does the aircraft stay in the trimmed turn? ----------
    import os
    import matplotlib.pyplot as plt
    os.makedirs("results", exist_ok=True)
    dt, t_end = 0.01, 120.0
    x, log = xt.copy(), []
    for t in np.arange(0.0, t_end, dt):
        log.append(np.concatenate([[t], x]))
        x = m6.rk4_step(x, ct, dt)
    log = np.array(log)
    t = log[:, 0]
    print(f"\nAfter {t_end:.0f} s in the trimmed turn: phi = {np.rad2deg(log[-1, 1 + m6.PHI]):.2f}°, "
          f"height change = {log[-1, 1 + m6.H] - log[0, 1 + m6.H]:.2f} m, "
          f"heading = {np.rad2deg(log[-1, 1 + m6.PSI]):.1f}°")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(log[:, 1 + m6.YE], log[:, 1 + m6.XN])
    ax[0].set(title="Ground track (trimmed turn, open loop)", xlabel="East [m]", ylabel="North [m]")
    ax[0].axis("equal")
    ax[1].plot(t, np.rad2deg(log[:, 1 + m6.PHI]), label="φ (bank)")
    ax[1].plot(t, np.rad2deg(log[:, 1 + m6.THETA]), label="θ (pitch)")
    ax[1].set(title="Attitude stays constant", xlabel="Time [s]", ylabel="[deg]")
    ax[1].legend()
    ax[2].plot(t, log[:, 1 + m6.H] - log[0, 1 + m6.H])
    ax[2].set(title="Height change", xlabel="Time [s]", ylabel="Δh [m]")
    for a in ax:
        a.grid()
    fig.suptitle("Coordinated right turn, 3°/s at 50 m/s: trim held without any controller")
    plt.tight_layout()
    plt.savefig("results/trim_turn.png", dpi=120)
    plt.show()
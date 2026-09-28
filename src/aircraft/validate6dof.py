"""
Validation of the 6-DOF model:
  1) trim point of the 3-DOF model is also an equilibrium of the 6-DOF model
  2) linearization: longitudinal block == 3-DOF, lateral block decoupled
  3) nonlinear: elevator doublet 3-DOF vs 6-DOF, aileron pulse (lateral motion)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import flightdynamics_6dof as m6
m6.INCLUDE_ALPHA_DOT = False     # the 3-DOF model has no Cm_alpha_dot -> compare like with like
from flightdynamics import calculate_derivatives
from trim import trim
from linearize import linearize

LON = [m6.U, m6.W, m6.Q, m6.THETA]
LAT = [m6.V_, m6.P, m6.R, m6.PHI]


def jacobian(fun, x0, c0):
    """Central-difference linearization of x_dot = fun(x, c)."""
    n, m = len(x0), len(c0)
    A, B = np.zeros((n, n)), np.zeros((n, m))
    for i in range(n):
        d = np.zeros(n); d[i] = 1e-6 * max(1.0, abs(x0[i]))
        A[:, i] = (fun(x0 + d, c0) - fun(x0 - d, c0)) / (2 * d[i])
    for j in range(m):
        d = np.zeros(m); d[j] = 1e-6 * max(1.0, abs(c0[j]))
        B[:, j] = (fun(x0, c0 + d) - fun(x0, c0 - d)) / (2 * d[j])
    return A, B


def print_modes(name, A):
    for lam in np.linalg.eigvals(A):
        if lam.imag < 0:
            continue
        wn = abs(lam)
        if abs(lam.imag) > 1e-9:
            print(f"  {name}: wn = {wn:6.3f} rad/s, zeta = {-lam.real / wn:5.3f}, period = {2*np.pi/lam.imag:5.1f} s")
        else:
            print(f"  {name}: real pole {lam.real:8.4f}  ->  time constant = {-1/lam.real:6.2f} s")


def simulate_6dof(x, c_trim, c_fun, t_end, dt=0.01):
    log = []
    for t in np.arange(0.0, t_end, dt):
        c = c_trim + c_fun(t)
        log.append(np.concatenate([[t], x]))
        x = x + m6.f(x, c) * dt
    return np.array(log)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    V0 = 50.0
    x3, u3 = trim(V0)
    x6 = m6.state_from_3dof(*x3)
    c6 = np.array([u3[0], 0.0, 0.0, u3[1]])          # [delta_e, delta_a, delta_r, T]

    # ---------- 1) equilibrium ----------
    xdot = m6.f(x6, c6)
    print("1) 6-DOF derivatives at the 3-DOF trim point (all ~0 except x_N_dot = V):")
    print("  ", np.round(xdot, 6))

    # ---------- 2) linearization ----------
    A6, B6 = jacobian(m6.f, x6, c6)
    A3, _ = linearize(x3, u3)
    print("\n2) Coupling between blocks (should be 0 in straight, level flight):",
          np.abs(A6[np.ix_(LON, LAT)]).max(), np.abs(A6[np.ix_(LAT, LON)]).max())
    print("Longitudinal modes, 3-DOF:"); print_modes("3-DOF", A3)
    print("Longitudinal modes, 6-DOF:"); print_modes("6-DOF", A6[np.ix_(LON, LON)])
    print("Lateral modes, 6-DOF:");      print_modes("lateral", A6[np.ix_(LAT, LAT)])

    # ---------- 3a) elevator doublet: 3-DOF vs 6-DOF ----------
    doublet = lambda t: np.deg2rad(2.0) * (float(1 <= t < 2) - float(2 <= t < 3))
    dt, t_end = 0.01, 30.0
    s3 = np.array(x3, dtype=float); th3 = []
    for t in np.arange(0.0, t_end, dt):
        th3.append(s3[2])
        s3 = s3 + np.array(calculate_derivatives(*s3, u3[0] + doublet(t), u3[1])) * dt
    r6 = simulate_6dof(x6.copy(), c6, lambda t: np.array([doublet(t), 0, 0, 0]), t_end, dt)
    print(f"\n3a) Elevator doublet: max |θ_3DOF - θ_6DOF| = "
          f"{np.rad2deg(np.max(np.abs(np.array(th3) - r6[:, 1 + m6.THETA]))):.2e} deg")

    # ---------- 3b) aileron pulse: lateral motion ----------
    pulse = lambda t: np.array([0.0, np.deg2rad(-5.0) if 1 <= t < 2 else 0.0, 0.0, 0.0])
    rl = simulate_6dof(x6.copy(), c6, pulse, 60.0, dt)
    t = rl[:, 0]
    V, beta = [], []
    for row in rl:
        Vi, _, bi = m6.air_data(row[1:])
        beta.append(bi)

    fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=False)
    ax[0, 0].plot(np.arange(0, t_end, dt), np.rad2deg(th3), label="3-DOF")
    ax[0, 0].plot(r6[:, 0], np.rad2deg(r6[:, 1 + m6.THETA]), "--", label="6-DOF")
    ax[0, 0].set(title="Elevator doublet: θ", xlabel="Time [s]", ylabel="θ [deg]")
    ax[0, 0].legend()
    ax[0, 1].plot(t, np.rad2deg(rl[:, 1 + m6.PHI]))
    ax[0, 1].set(title="Aileron pulse: bank angle φ", xlabel="Time [s]", ylabel="φ [deg]")
    ax[0, 2].plot(t, np.rad2deg(rl[:, 1 + m6.P]), label="p (roll)")
    ax[0, 2].plot(t, np.rad2deg(rl[:, 1 + m6.R]), label="r (yaw)")
    ax[0, 2].set(title="Aileron pulse: rates", xlabel="Time [s]", ylabel="[deg/s]", xlim=(0, 10))
    ax[0, 2].legend()
    ax[1, 0].plot(t, np.rad2deg(beta))
    ax[1, 0].set(title="Aileron pulse: sideslip β", xlabel="Time [s]", ylabel="β [deg]", xlim=(0, 10))
    ax[1, 1].plot(t, np.rad2deg(rl[:, 1 + m6.PSI]))
    ax[1, 1].set(title="Aileron pulse: heading ψ", xlabel="Time [s]", ylabel="ψ [deg]")
    ax[1, 2].plot(rl[:, 1 + m6.YE], rl[:, 1 + m6.XN])
    ax[1, 2].set(title="Aileron pulse: ground track", xlabel="East [m]", ylabel="North [m]")
    ax[1, 2].axis("equal")
    for a in ax.flat:
        a.grid()
    fig.suptitle("6-DOF validation (V = 50 m/s, open loop)")
    plt.tight_layout()
    plt.savefig("results/validate_6dof.png", dpi=120)
    plt.show()
"""
LQI controller for the longitudinal motion at one operating point.

    Plant (linear):   dx_dot = A dx + B du          dx = [dV, dgamma, dtheta, q]
    Tracked outputs:  y = C dx = [dV, dtheta]       du = [d_delta_e, dT]
    Integrators:      z_dot = r - y
    Control law:      u = u0 - Kx (x - x0) - Kz z
"""
import os
import numpy as np
import control as ct
import matplotlib.pyplot as plt
import c172_params as p
from flightdynamics import calculate_derivatives
from trim import trim
from linearize import linearize

# ============================================================
# Tracked outputs: V and theta
# ============================================================
C = np.array([[1.0, 0.0, 0.0, 0.0],      # y1 = V
              [0.0, 0.0, 1.0, 0.0]])     # y2 = theta

# ============================================================
# Bryson's rule: maximum acceptable deviations
# ============================================================
x_max = [3.0,                  # dV      [m/s]
         np.deg2rad(10.0),     # dgamma  [rad]
         np.deg2rad(5.0),      # dtheta  [rad]
         np.deg2rad(10.0),     # q       [rad/s]
         5.0,                  # z_V     [m]
         np.deg2rad(3.0)]      # z_theta [rad*s]
u_max = [np.deg2rad(10.0),     # d_delta_e [rad]
         1000.0]               # dT        [N]

Q = np.diag(1.0 / np.array(x_max)**2)
R = np.diag(1.0 / np.array(u_max)**2)


def design_lqi(A, B):
    """Augment the plant with integrators and solve the LQR problem."""
    n, m = A.shape[0], B.shape[1]
    k = C.shape[0]

    A_aug = np.block([[A, np.zeros((n, k))],
                      [-C, np.zeros((k, k))]])
    B_aug = np.vstack([B, np.zeros((k, m))])

    rank = np.linalg.matrix_rank(ct.ctrb(A_aug, B_aug))
    if rank < n + k:
        raise RuntimeError(f"Augmented system not controllable (rank {rank} < {n + k})")

    K_aug, _, poles = ct.lqr(A_aug, B_aug, Q, R)
    Kx, Kz = K_aug[:, :n], K_aug[:, n:]
    return Kx, Kz, poles


def lqi_control(x, x0, u0, z, r, Kx, Kz, dt, anti_windup=True):
    """
    One controller step.
    Returns the saturated input u = [delta_e, T] and the updated integrator z.
    """
    y = C @ x
    error = r - y
    u_cmd = u0 - Kx @ (x - x0) - Kz @ z

    u_min = np.array([-p.delta_e_max, p.T_min])
    u_max_abs = np.array([p.delta_e_max, p.T_max])
    u = np.clip(u_cmd, u_min, u_max_abs)

    # Anti-windup: freeze the integrators if a saturated input
    # would be pushed further into its limit by integrating.
    freeze = False
    if anti_windup:
        for j in range(len(u)):
            if u[j] != u_cmd[j]:
                push_direction = np.sign(u_cmd[j] - u[j])       # +1 upper limit, -1 lower limit
                integrator_effect = np.sign(-Kz[j] @ error)      # where integrating moves u_j
                if integrator_effect == push_direction:
                    freeze = True
    if not freeze:
        z = z + error * dt

    return u, z


def simulate_lqi(Kx, Kz, V0=50.0, dV=5.0, dtheta_deg=3.0, t_step=2.0, t_end=60.0, dt=0.01, gust=None):
    """Nonlinear simulation: steps on V and theta at the same time."""
    x0, u0 = trim(V0)
    x = x0.copy()
    z = np.zeros(2)

    log = {k: [] for k in ["t", "V", "V_set", "theta", "theta_set", "gamma", "delta_e", "T"]}

    for t in np.arange(0.0, t_end, dt):
        stepped = t >= t_step
        r = np.array([x0[0] + (dV if stepped else 0.0),
                      x0[2] + (np.deg2rad(dtheta_deg) if stepped else 0.0)])

        u, z = lqi_control(x, x0, u0, z, r, Kx, Kz, dt)

        x_dot = np.array(calculate_derivatives(*x, *u, gust(t) if gust else 0.0))
        x = x + x_dot * dt

        for key, val in zip(log, [t, x[0], r[0], np.rad2deg(x[2]), np.rad2deg(r[1]),
                                  np.rad2deg(x[1]), np.rad2deg(u[0]), u[1]]):
            log[key].append(val)

    return {k: np.array(v) for k, v in log.items()}


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    V0 = 50.0
    A, B = linearize(*trim(V0))
    Kx, Kz, poles = design_lqi(A, B)

    print(f"LQI at V = {V0} m/s")
    print("Kx (rows: delta_e, T | cols: V, gamma, theta, q) =\n", Kx)
    print("Kz (rows: delta_e, T | cols: z_V, z_theta) =\n", Kz)
    print("\nClosed-loop poles:")
    for s in poles:
        if s.imag >= 0:
            wn = abs(s)
            print(f"  {s.real:8.3f} {s.imag:+8.3f}j   wn = {wn:6.3f}   zeta = {-s.real / wn:5.2f}")

    res = simulate_lqi(Kx, Kz, V0=V0)

    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    ax[0, 0].plot(res["t"], res["V"]); ax[0, 0].plot(res["t"], res["V_set"], "k--")
    ax[0, 1].plot(res["t"], res["theta"]); ax[0, 1].plot(res["t"], res["theta_set"], "k--")
    ax[1, 0].plot(res["t"], res["delta_e"])
    ax[1, 1].plot(res["t"], res["T"]); ax[1, 1].axhline(p.T_max, color="r", ls=":")
    ax[0, 0].set(ylabel="V [m/s]", title="Airspeed")
    ax[0, 1].set(ylabel="θ [deg]", title="Pitch angle")
    ax[1, 0].set(ylabel="δe [deg]", xlabel="Time [s]", title="Elevator")
    ax[1, 1].set(ylabel="T [N]", xlabel="Time [s]", title="Thrust")
    for a in ax.flat:
        a.grid()
    fig.suptitle("LQI, nonlinear model: V +5 m/s and θ +3° at the same time")
    plt.tight_layout()
    plt.savefig("results/lqi_step.png", dpi=120)
    plt.show()
"""
Lateral autopilot as ONE MIMO controller: gain-scheduled LQI for bank angle + sideslip.

    States (lateral part of the 6-DOF state): x_lat = [v, p, r, phi]
    Inputs:                                   [delta_a, delta_r]
    Tracked outputs:                          y = [phi, beta]     (targets: phi_set, 0)
    Integrators:                              z_dot = y_ref - y
    Control law:  c = c0 - Kx(V) (x_lat - x_ref) - Kz(V) z

x_ref is the state of the intended coordinated turn (same idea as r_ref in the PID):
    p_ref = -psi_dot*sin(theta),  r_ref = psi_dot*cos(phi)*cos(theta),  psi_dot = g*tan(phi_set)/V
Kx, Kz are interpolated at the measured airspeed (gain scheduling).
"""
import os
import numpy as np
import control as ct
import matplotlib.pyplot as plt
import aircraft
from aircraft import p
import flightdynamics_6dof as m6
from trim_6dof import trim_6dof, linearize_6dof
from autopilot_lon import LongitudinalAutopilot
from autopilot_lat import LateralAutopilot, bank_profile, summary

LAT = [m6.V_, m6.P, m6.R, m6.PHI]
INPUTS = [m6.DA, m6.DR]
V_GRID = np.arange(30.0, 65.01, 5.0)

# Bryson weights: maximum acceptable deviations
X_MAX = [1.0, np.deg2rad(20.0), np.deg2rad(5.0), np.deg2rad(3.0),   # v [m/s], p, r, phi
         np.deg2rad(6.0), np.deg2rad(2.0)]                           # z_phi [rad·s], z_beta [rad·s]
U_MAX = [np.deg2rad(10.0), np.deg2rad(10.0)]                        # delta_a, delta_r


def design_point(V, h=1000.0):
    """Trim, linearize and solve the lateral LQI problem at one airspeed."""
    x0, c0 = trim_6dof(V, h=h)
    A, B = linearize_6dof(x0, c0)
    A4, B2 = A[np.ix_(LAT, LAT)], B[np.ix_(LAT, INPUTS)]
    C = np.array([[0, 0, 0, 1],                   # phi
                  [1 / V, 0, 0, 0]])              # beta ≈ v / V
    A_aug = np.block([[A4, np.zeros((4, 2))], [-C, np.zeros((2, 2))]])
    B_aug = np.vstack([B2, np.zeros((2, 2))])
    if np.linalg.matrix_rank(ct.ctrb(A_aug, B_aug)) < 6:
        raise RuntimeError(f"not controllable at V = {V}")
    Q = np.diag(1 / np.array(X_MAX)**2)
    R = np.diag(1 / np.array(U_MAX)**2)
    K, _, poles = ct.lqr(A_aug, B_aug, Q, R)
    return c0[INPUTS], K[:, :4], K[:, 4:], poles


def build_schedule():
    rows = [design_point(V) for V in V_GRID]
    return {k: np.array([r[i] for r in rows]) for i, k in enumerate(["c0", "Kx", "Kz"])}


def interp(table, key, V):
    data = table[key]
    flat = data.reshape(len(V_GRID), -1)
    vals = [np.interp(V, V_GRID, flat[:, i]) for i in range(flat.shape[1])]
    return np.array(vals).reshape(data.shape[1:])


class LQIAutopilotLat:
    phi_rate_max = LateralAutopilot.phi_rate_max  # same command shaping as the PID -> fair comparison
    phi_max = LateralAutopilot.phi_max

    def __init__(self, schedule, x_trim):
        self.s = schedule
        self.z = np.zeros(2)
        self.phi_set = x_trim[m6.PHI]

    def step(self, x, phi_cmd, dt):
        V, _, beta = m6.air_data(x)
        target = np.clip(phi_cmd, -self.phi_max, self.phi_max)
        max_step = self.phi_rate_max * dt
        self.phi_set += np.clip(target - self.phi_set, -max_step, max_step)

        # reference state: steady coordinated turn at phi_set
        psi_dot = p.g * np.tan(self.phi_set) / V
        th = x[m6.THETA]
        x_ref = np.array([0.0, -psi_dot * np.sin(th),
                          psi_dot * np.cos(self.phi_set) * np.cos(th), self.phi_set])

        c0 = interp(self.s, "c0", V)
        Kx, Kz = interp(self.s, "Kx", V), interp(self.s, "Kz", V)
        e = np.array([self.phi_set - x[m6.PHI], 0.0 - beta])
        u_cmd = c0 - Kx @ (x[LAT] - x_ref) - Kz @ self.z
        lim = np.array([p.delta_a_max, p.delta_r_max])
        u = np.clip(u_cmd, -lim, lim)
        if np.all(u == u_cmd):                    # anti-windup: freeze when saturated
            self.z += e * dt
        return u[0], u[1]


def simulate(make_lat, V0=50.0, h0=1000.0, t_end=60.0, dt=0.02):
    x, c0 = trim_6dof(V0, h=h0)
    lon = LongitudinalAutopilot(x, c0)
    lat = make_lat(x, c0)
    keys = ["t", "phi", "phi_set", "beta", "psi", "h", "da", "dr"]
    log = {k: [] for k in keys}
    for t in np.arange(0.0, t_end, dt):
        c, _ = lon.step(x, h0, V0, dt)
        c[m6.DA], c[m6.DR] = lat.step(x, bank_profile(t), dt)
        x = m6.rk4_step(x, c, dt)
        _, _, beta = m6.air_data(x)
        vals = [t, np.rad2deg(x[m6.PHI]), np.rad2deg(lat.phi_set), np.rad2deg(beta),
                np.rad2deg(x[m6.PSI]), x[m6.H], np.rad2deg(c[m6.DA]), np.rad2deg(c[m6.DR])]
        for k, v in zip(keys, vals):
            log[k].append(v)
    return {k: np.array(v) for k, v in log.items()}


if __name__ == "__main__":
    aircraft.load("c172")
    os.makedirs("results", exist_ok=True)
    np.set_printoptions(precision=3, suppress=True)
    schedule = build_schedule()
    _, Kx, Kz, poles = design_point(50.0)
    print("Lateral LQI at 50 m/s  (rows: delta_a, delta_r)")
    print("Kx [v, p, r, phi] =\n", Kx, "\nKz [z_phi, z_beta] =\n", Kz)
    print("closed-loop poles:", np.round(poles, 2))

    pid = lambda x, c: LateralAutopilot(x, c)
    lqi = lambda x, c: LQIAutopilotLat(schedule, x)
    runs = {(V, n): simulate(f, V0=V) for V in [30.0, 50.0, 65.0] for n, f in [("PID", pid), ("LQI", lqi)]}

    keys = ["max phi overshoot [deg]", "phi error in turn [deg]", "max |beta| [deg]", "max altitude loss [m]"]
    print(f"\n{'':26s}" + "".join(f"{f'{V:.0f} m/s {n}':>12s}" for V, n in runs))
    for k in keys:
        print(f"{k:26s}" + "".join(f"{summary(r)[k]:12.2f}" for r in runs.values()))

    fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
    for j, V in enumerate([30.0, 50.0, 65.0]):
        for n, col in [("PID", "#2a6fdb"), ("LQI", "#e8782a")]:
            r = runs[(V, n)]
            ax[0, j].plot(r["t"], r["phi"], color=col, label=n)
            ax[1, j].plot(r["t"], r["beta"], color=col, label=n)
        ax[0, j].plot(r["t"], r["phi_set"], "k--", lw=1, label="φ_set")
        ax[0, j].set_title(f"{V:.0f} m/s – bank angle φ [deg]")
        ax[1, j].set_title(f"{V:.0f} m/s – sideslip β [deg]")
        ax[1, j].set_xlabel("Time [s]")
    for a in ax.flat:
        a.grid(); a.legend(fontsize=8)
    fig.suptitle("C172 lateral autopilot: PID (fixed gains) vs. LQI (scheduled over V) – bank 20° and back")
    plt.tight_layout()
    plt.savefig("results/autopilot_lat_pid_vs_lqi.png", dpi=120)
"""
Longitudinal autopilot as ONE MIMO controller: gain-scheduled LQI for altitude + airspeed.

    States (longitudinal part of the 6-DOF state): x_lon = [u, w, q, theta, h]
    Inputs:                                       [delta_e, T]
    Tracked outputs:                              y = [V, h]
    Integrators:                                  z_dot = r - y
    Control law:  c = c0(V_set) - Kx(V) (x_lon - x0(V_set, h_set)) - Kz(V) z

Gain scheduling (same idea as in Phase 3):
    Kx, Kz   interpolated at the MEASURED airspeed
    x0, c0   interpolated at the COMMANDED airspeed (feedforward of the trim point)
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

LON = [m6.U, m6.W, m6.Q, m6.THETA, m6.H]
INPUTS = [m6.DE, m6.TH]
V_GRID = np.arange(30.0, 65.01, 5.0)

# Bryson weights: maximum acceptable deviations
X_MAX = [2.0, 2.0, np.deg2rad(10.0), np.deg2rad(5.0), 10.0,   # du, dw, q, theta, h
         5.0, 50.0]                                            # z_V [m], z_h [m·s]
U_MAX = [np.deg2rad(10.0), 1000.0]                             # delta_e [rad], T [N]


def design_point(V, h=1000.0):
    """Trim, linearize and solve the LQI problem at one airspeed."""
    x0, c0 = trim_6dof(V, h=h)
    A, B = linearize_6dof(x0, c0)
    A5, B2 = A[np.ix_(LON, LON)], B[np.ix_(LON, INPUTS)]
    u0, w0 = x0[m6.U], x0[m6.W]
    C = np.array([[u0 / V, w0 / V, 0, 0, 0],       # dV ≈ (u0 du + w0 dw) / V
                  [0, 0, 0, 0, 1]])                 # dh
    A_aug = np.block([[A5, np.zeros((5, 2))], [-C, np.zeros((2, 2))]])
    B_aug = np.vstack([B2, np.zeros((2, 2))])
    if np.linalg.matrix_rank(ct.ctrb(A_aug, B_aug)) < 7:
        raise RuntimeError(f"not controllable at V = {V}")
    Q = np.diag(1 / np.array(X_MAX)**2)
    R = np.diag(1 / np.array(U_MAX)**2)
    K, _, poles = ct.lqr(A_aug, B_aug, Q, R)
    return x0[LON], c0, K[:, :5], K[:, 5:], poles


def build_schedule():
    rows = [design_point(V) for V in V_GRID]
    return {"x0": np.array([r[0] for r in rows]), "c0": np.array([r[1] for r in rows]),
            "Kx": np.array([r[2] for r in rows]), "Kz": np.array([r[3] for r in rows])}


def interp(table, key, V):
    data = table[key]
    flat = data.reshape(len(V_GRID), -1)
    vals = [np.interp(V, V_GRID, flat[:, i]) for i in range(flat.shape[1])]
    return np.array(vals).reshape(data.shape[1:])


class LQIAutopilotLon:
    def __init__(self, schedule):
        self.s = schedule
        self.z = np.zeros(2)

    use_feedforward = False
    turn_ff = None                                     # optional TurnFeedforward (Phase 6, step 4)

    def step(self, x, h_set, V_set, dt, h_dot_ref=0.0):
        V, _, _ = m6.air_data(x)
        x0 = interp(self.s, "x0", V_set)
        x0[4] = h_set                                  # target altitude replaces trim altitude
        c0 = interp(self.s, "c0", V_set)
        Kx, Kz = interp(self.s, "Kx", V), interp(self.s, "Kz", V)

        e = np.array([V_set - V, h_set - x[m6.H]])
        if self.use_feedforward:
            # planned climb: reference state is a steady climb (nose up by gamma, more thrust)
            gamma_ref = np.arcsin(np.clip(h_dot_ref / V_set, -0.5, 0.5))
            x0[3] += gamma_ref
            c0[m6.TH] += p.m * p.g * np.sin(gamma_ref)
        if self.turn_ff:
            # level turn: reference state and inputs from the turn trim at the measured bank
            tf = self.turn_ff(x[m6.PHI])
            x0 += [tf["du"], tf["dw"], tf["q"], tf["dtheta"], 0.0]
            c0[m6.DE] += tf["dde"]
            c0[m6.TH] += tf["dT"]
        u_cmd = c0[INPUTS] - Kx @ (x[LON] - x0) - Kz @ self.z
        lo = np.array([p.delta_e_min, p.T_min])
        hi = np.array([p.delta_e_max_6dof, p.T_max])
        u = np.clip(u_cmd, lo, hi)

        # anti-windup: freeze integrators if a saturated input would be driven further
        freeze = any(u[j] != u_cmd[j] and np.sign(-Kz[j] @ e) == np.sign(u_cmd[j] - u[j])
                     for j in range(2))
        if not freeze:
            self.z += e * dt

        c = c0.copy()
        c[m6.DE], c[m6.TH] = u
        return c, None


def climb_profile(t, t_start, dh, rate, t_ramp=3.0):
    """Smooth climb: climb rate ramps up over t_ramp, holds, ramps down. Returns (h offset, h_dot)."""
    t_cruise = dh / rate - t_ramp                    # time at full climb rate
    tau = t - t_start
    if tau <= 0:
        return 0.0, 0.0
    if tau < t_ramp:                                 # ramp up
        return 0.5 * rate / t_ramp * tau**2, rate * tau / t_ramp
    if tau < t_ramp + t_cruise:                      # constant climb rate
        return 0.5 * rate * t_ramp + rate * (tau - t_ramp), rate
    tau2 = tau - t_ramp - t_cruise
    if tau2 < t_ramp:                                # ramp down
        return (0.5 * rate * t_ramp + rate * t_cruise
                + rate * tau2 - 0.5 * rate / t_ramp * tau2**2), rate * (1 - tau2 / t_ramp)
    return dh, 0.0


def shaped_refs(h0, V0, dh=100.0, dV=5.0, climb_rate=5.0, accel=1.0):
    """Same smooth reference for both controllers: returns (h_set, V_set, h_dot_ref)."""
    def refs(t):
        h_off, h_dot = climb_profile(t, 5.0, dh, climb_rate)
        V = V0 + np.clip((t - 60.0) * accel, 0.0, dV)
        return h0 + h_off, V, h_dot
    return refs


def simulate(name, make_controller, V0=50.0, h0=1000.0, t_end=120.0, dt=0.02, refs=None):
    aircraft.load(name)
    x, c0 = trim_6dof(V0, h=h0)
    ctrl = make_controller(x, c0)
    refs = refs or shaped_refs(h0, V0)
    log = {k: [] for k in ["t", "h", "h_set", "V", "V_set", "de", "T"]}
    for t in np.arange(0.0, t_end, dt):
        h_set, V_set, h_dot_ref = refs(t)
        c, _ = ctrl.step(x, h_set, V_set, dt, h_dot_ref)
        x = m6.rk4_step(x, c, dt)
        V, _, _ = m6.air_data(x)
        for k, v in zip(log, [t, x[m6.H], h_set, V, V_set, np.rad2deg(c[m6.DE]), c[m6.TH]]):
            log[k].append(v)
    aircraft.load(aircraft.DEFAULT)
    return {k: np.array(v) for k, v in log.items()}


def pid_factory_ff(ff):
    def make(x, c0):
        ctrl = LongitudinalAutopilot(x, c0)
        ctrl.use_feedforward = ff
        return ctrl
    return make


def lqi_factory_for(name, ff=False):
    aircraft.load(name)
    schedule = build_schedule()
    def make(x, c0):
        ctrl = LQIAutopilotLon(schedule)
        ctrl.use_feedforward = ff
        return ctrl
    return make


def summary(r):
    climb = (r["t"] >= 5) & (r["t"] < 60)            # climb phase incl. levelling off
    accel = r["t"] >= 60
    return {
        "max |h err| climb [m]": np.abs(r["h"] - r["h_set"])[climb].max(),
        "max |V err| climb [m/s]": np.abs(r["V"] - r["V_set"])[climb].max(),
        "max |h err| speed change [m]": np.abs(r["h"] - r["h_set"])[accel].max(),
        "max |V err| speed change [m/s]": np.abs(r["V"] - r["V_set"])[accel].max(),
        "elevator range [deg]": r["de"].max() - r["de"].min(),
    }


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    np.set_printoptions(precision=3, suppress=True)

    # Show the gains at one design point
    aircraft.load("c172")
    _, _, Kx, Kz, poles = design_point(50.0)
    print("C172 LQI at 50 m/s  (rows: delta_e, T)")
    print("Kx  [u, w, q, theta, h] =\n", Kx)
    print("Kz  [z_V, z_h] =\n", Kz)

    runs = {}
    for name in ["c172", "c182"]:
        for ff in [False, True]:
            tag = "+FF" if ff else ""
            runs[(name, "PID" + tag)] = simulate(name, pid_factory_ff(ff))
            runs[(name, "LQI" + tag)] = simulate(name, lqi_factory_for(name, ff))

    keys = list(summary(runs[("c172", "PID")]).keys())
    print(f"\n{'':32s}" + "".join(f"{n.upper()+' '+c:>13s}" for n, c in runs))
    for k in keys:
        print(f"{k:32s}" + "".join(f"{summary(r)[k]:13.2f}" for r in runs.values()))

    fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    styles = {"PID": ("#2a6fdb", ":"), "PID+FF": ("#2a6fdb", "-"),
              "LQI": ("#e8782a", ":"), "LQI+FF": ("#e8782a", "-")}
    for (name, ctrl), r in runs.items():
        if name != "c172":
            continue                                  # plot C172; C182 is in the table
        col, ls = styles[ctrl]
        ax[0, 0].plot(r["t"], r["h"] - r["h_set"], ls, color=col, label=ctrl)
        ax[0, 1].plot(r["t"], r["V"] - r["V_set"], ls, color=col, label=ctrl)
        ax[1, 0].plot(r["t"], r["de"], ls, color=col, label=ctrl)
        ax[1, 1].plot(r["t"], r["T"], ls, color=col, label=ctrl)
    titles = ["Altitude tracking error h - h_set [m]", "Speed tracking error V - V_set [m/s]",
              "Elevator δe [deg]", "Thrust T [N]"]
    for a, tt in zip(ax.flat, titles):
        a.set_title(tt); a.grid(); a.legend(fontsize=8)
        a.axvspan(5, 28, color="#999999", alpha=0.12)
        a.axvspan(60, 65, color="#999999", alpha=0.12)
    for a in ax[1]:
        a.set_xlabel("Time [s]")
    fig.suptitle("C172: without (dotted) vs. with climb-rate feedforward (solid) – "
                 "climb +100 m (grey 5–28 s), then +5 m/s (grey 60–65 s)")
    plt.tight_layout()
    plt.savefig("results/autopilot_lon_feedforward.png", dpi=120)
    plt.show()
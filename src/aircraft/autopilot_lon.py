"""
Longitudinal autopilot on the 6-DOF model: ALTITUDE HOLD + SPEED HOLD (PID cascade).

    h_set ──▶ altitude loop (slow) ──θ_cmd──▶ pitch loop (fast) ──▶ elevator
    V_set ──▶ speed loop (PI) ─────────────────────────────────────▶ thrust

Lateral controls stay at their trim values (wings level, no turn) in this step.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import aircraft
from aircraft import p
import flightdynamics_6dof as m6
from trim_6dof import trim_6dof
from turn_ff import ZERO


class LongitudinalAutopilot:
    # Pitch attitude loop (inner) – gains from pitch_design.py
    Kp_th, Kd_th, Ki_th = 1.5, 0.2, 0.6
    # Altitude loop (outer): theta_cmd = theta_trim + Kh*e_h + Ki_h*∫e_h - K_vs*h_dot
    Kh, Ki_h, K_vs = 0.006, 0.0003, 0.012        # [rad/m], [rad/(m·s)], [rad/(m/s)]  (tuned)
    I_h_band = 10.0                              # integrate altitude error only within ±10 m
    theta_cmd_max = np.deg2rad(8.0)              # pitch command limit (relative to trim)
    theta_rate_max = np.deg2rad(3.0)             # pitch command rate limit [rad/s] (comfort)
    # Speed loop: PI with gains from the desired closed-loop dynamics (see Phase 3)
    wn_v, zeta_v = 0.5, 0.8

    def __init__(self, x_trim, c_trim):
        self.theta_trim = x_trim[m6.THETA]
        self.c_trim = c_trim.copy()
        self.Kp_v = 2 * self.zeta_v * self.wn_v * p.m
        self.Ki_v = self.wn_v**2 * p.m
        self.I_th = self.I_h = self.I_v = 0.0
        self.theta_set = self.theta_trim            # rate-limited pitch command (memory)

    use_feedforward = False
    # gain scheduling: control surfaces get weaker at low speed (force ~ V²) -> scale the
    # elevator gains with (V_ref/V)², the gains were tuned at V_ref
    schedule_gains = True
    V_ref, sched_exp = 50.0, 2.0
    turn_ff = None                               # optional TurnFeedforward (Phase 6, step 4)

    def step(self, x, h_set, V_set, dt, h_dot_ref=0.0):
        V, alpha, _ = m6.air_data(x)
        h, theta, q = x[m6.H], x[m6.THETA], x[m6.Q]
        # climb rate from the kinematics (cheaper than evaluating the full model)
        u, v, w, phi = x[m6.U], x[m6.V_], x[m6.W], x[m6.PHI]
        h_dot = u * np.sin(theta) - v * np.sin(phi) * np.cos(theta) - w * np.cos(phi) * np.cos(theta)

        # turn feedforward: extra pitch, elevator, thrust and the pitch rate of a level turn
        tf = self.turn_ff(x[m6.PHI]) if self.turn_ff else ZERO

        # ---- outer loop: altitude -> pitch command ----
        e_h = h_set - h
        # feedforward: planned climb -> flight path angle gamma_ref (the nose must rise by that much)
        gamma_ref = np.arcsin(np.clip(h_dot_ref / V_set, -0.5, 0.5)) if self.use_feedforward else 0.0
        hd_ref = h_dot_ref if self.use_feedforward else 0.0
        theta_cmd = gamma_ref + self.Kh * e_h + self.Ki_h * self.I_h - self.K_vs * (h_dot - hd_ref)
        theta_cmd_sat = np.clip(theta_cmd, -self.theta_cmd_max, self.theta_cmd_max)
        # anti-windup: integrate only near the target altitude and when not limited
        if theta_cmd_sat == theta_cmd and abs(e_h) < self.I_h_band:
            self.I_h += e_h * dt
        # rate limiter: the pitch command may change by at most theta_rate_max per second
        target = self.theta_trim + theta_cmd_sat + tf["dtheta"]
        max_step = self.theta_rate_max * dt
        self.theta_set += np.clip(target - self.theta_set, -max_step, max_step)
        theta_set = self.theta_set

        # ---- inner loop: pitch attitude -> elevator ----
        e_th = theta_set - theta
        s = (self.V_ref / V) ** self.sched_exp if self.schedule_gains else 1.0
        de_cmd = (self.c_trim[m6.DE] + tf["dde"]
                  + s * (-self.Kp_th * e_th - self.Ki_th * self.I_th + self.Kd_th * (q - tf["q"])))      # damp only the q that does NOT belong to the turn
        de = np.clip(de_cmd, p.delta_e_min, p.delta_e_max_6dof)
        if de == de_cmd:
            self.I_th += e_th * dt

        # ---- speed loop: airspeed -> thrust ----
        e_v = V_set - V
        T_ff = p.m * p.g * np.sin(gamma_ref)       # feedforward: extra thrust for climbing
        T_cmd = self.c_trim[m6.TH] + T_ff + tf["dT"] + self.Kp_v * e_v + self.Ki_v * self.I_v
        T = np.clip(T_cmd, p.T_min, p.T_max)
        if T == T_cmd or np.sign(e_v) != np.sign(T_cmd - T):
            self.I_v += e_v * dt

        c = self.c_trim.copy()
        c[m6.DE], c[m6.TH] = de, T
        return c, theta_set


def simulate(name, V0=50.0, h0=1000.0, dh_step=100.0, dV_step=5.0, t_end=120.0, dt=0.01,
             gust=None):
    aircraft.load(name)
    x, c0 = trim_6dof(V0, h=h0)
    ap = LongitudinalAutopilot(x, c0)
    log = {k: [] for k in ["t", "h", "h_set", "V", "V_set", "theta", "theta_set", "de", "T"]}
    for t in np.arange(0.0, t_end, dt):
        h_set = h0 + (dh_step if t >= 5.0 else 0.0)
        V_set = V0 + (dV_step if t >= 60.0 else 0.0)
        c, theta_set = ap.step(x, h_set, V_set, dt)
        wind = (0.0, 0.0, -gust(t)) if gust else (0.0, 0.0, 0.0)   # NED: up-gust = negative z
        x = m6.rk4_step(x, c, dt, wind)
        V, _, _ = m6.air_data(x)
        for k, v in zip(log, [t, x[m6.H], h_set, V, V_set, np.rad2deg(x[m6.THETA]),
                              np.rad2deg(theta_set), np.rad2deg(c[m6.DE]), c[m6.TH]]):
            log[k].append(v)
    aircraft.load(aircraft.DEFAULT)
    return {k: np.array(v) for k, v in log.items()}


def metrics(r, t0, t1, key, target_key):
    m = (r["t"] >= t0) & (r["t"] < t1)
    err = r[key][m] - r[target_key][m]
    final = r[target_key][m][-1]
    start = r[key][m][0]
    step = final - start
    overshoot = max(0.0, np.max((r[key][m] - final) * np.sign(step))) if step else 0.0
    band = 0.02 * abs(step) if step else 0.5
    outside = np.where(np.abs(err) > band)[0]
    settle = r["t"][m][outside[-1]] - t0 if len(outside) else 0.0
    return overshoot, settle


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    runs = {name: simulate(name) for name in ["c172", "c182"]}

    for name, r in runs.items():
        os_h, ts_h = metrics(r, 5, 60, "h", "h_set")
        os_v, ts_v = metrics(r, 60, 120, "V", "V_set")
        dh_speed = np.abs(r["h"][r["t"] >= 60] - r["h_set"][r["t"] >= 60]).max()
        print(f"{name}: altitude +100 m -> overshoot {os_h:4.1f} m, settled (±2 m) after {ts_h:4.1f} s | "
              f"speed +5 m/s -> overshoot {os_v:4.2f} m/s, settled after {ts_v:4.1f} s, "
              f"max altitude deviation during speed change {dh_speed:4.1f} m")

    fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for name, r in runs.items():
        ax[0, 0].plot(r["t"], r["h"], label=name.upper())
        ax[0, 1].plot(r["t"], r["V"], label=name.upper())
        ax[0, 2].plot(r["t"], r["theta"], label=name.upper())
        ax[1, 0].plot(r["t"], r["de"], label=name.upper())
        ax[1, 1].plot(r["t"], r["T"], label=name.upper())
        ax[1, 2].plot(r["t"], r["theta_set"], label=f"θ_set {name.upper()}")
    r = runs["c172"]
    ax[0, 0].plot(r["t"], r["h_set"], "k--", lw=1, label="setpoint")
    ax[0, 1].plot(r["t"], r["V_set"], "k--", lw=1, label="setpoint")
    titles = [["Altitude h [m]", "Airspeed V [m/s]", "Pitch θ [deg]"],
              ["Elevator δe [deg]", "Thrust T [N]", "Pitch command from altitude loop [deg]"]]
    for i in range(2):
        for j in range(3):
            ax[i, j].set_title(titles[i][j])
            ax[i, j].grid()
            ax[i, j].legend(fontsize=8)
    for a in ax[1]:
        a.set_xlabel("Time [s]")
    fig.suptitle("Longitudinal autopilot (6-DOF): +100 m altitude at t = 5 s, +5 m/s speed at t = 60 s")
    plt.tight_layout()
    plt.savefig("results/autopilot_lon.png", dpi=120)
    plt.show()
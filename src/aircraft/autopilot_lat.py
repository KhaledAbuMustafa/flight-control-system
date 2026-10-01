"""
Lateral autopilot on the 6-DOF model (PID): BANK-ANGLE HOLD + YAW DAMPER / TURN COORDINATION.

    phi_set ──▶ rate limiter ──▶ bank loop (PID on phi, D on p) ──▶ aileron
    r, beta ──▶ yaw damper (only the deviation from the intended turn rate) ──▶ rudder

Sign conventions (see c172_params.py):
    +delta_a = right aileron down -> rolls LEFT     -> a "minus" is needed to roll right
    +delta_r = trailing edge left -> nose LEFT
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import aircraft
from aircraft import p
import flightdynamics_6dof as m6
from trim_6dof import trim_6dof
from autopilot_lon import LongitudinalAutopilot


class LateralAutopilot:
    # Bank loop: gains from the linear exploration at 50 m/s (Dutch roll zeta ≈ 0.95 together with Kr)
    Kp_phi, Ki_phi, Kd_phi = 1.0, 0.1, 0.1       # [rad/rad], [rad/(rad·s)], [rad/(rad/s)]
    # Yaw damper + coordination
    Kr, Kb = 0.5, 1.0                            # [rad/(rad/s)], [rad/rad]  (Kb from the sim sweep)
    phi_rate_max = np.deg2rad(10.0)              # bank command rate limit [rad/s]
    phi_max = np.deg2rad(30.0)                   # bank command limit
    # gain scheduling: aileron/rudder effect ~ V² -> scale all gains with (V_ref/V)²
    schedule_gains = True
    V_ref, sched_exp = 50.0, 2.0

    def __init__(self, x_trim, c_trim):
        self.c_trim = c_trim.copy()
        self.I_phi = 0.0
        self.phi_set = x_trim[m6.PHI]            # rate-limited bank command (memory)

    def step(self, x, phi_cmd, dt):
        V, _, beta = m6.air_data(x)
        phi, p_, r = x[m6.PHI], x[m6.P], x[m6.R]

        # ---- bank command: limit + rate limiter (smooth roll-in) ----
        target = np.clip(phi_cmd, -self.phi_max, self.phi_max)
        max_step = self.phi_rate_max * dt
        self.phi_set += np.clip(target - self.phi_set, -max_step, max_step)

        # ---- bank loop -> aileron ----
        e_phi = self.phi_set - phi
        s = (self.V_ref / V) ** self.sched_exp if self.schedule_gains else 1.0
        da_cmd = (self.c_trim[m6.DA]
                  + s * (-self.Kp_phi * e_phi - self.Ki_phi * self.I_phi + self.Kd_phi * p_))
        da = np.clip(da_cmd, -p.delta_a_max, p.delta_a_max)
        if da == da_cmd:                         # anti-windup: integrate only when not saturated
            self.I_phi += e_phi * dt

        # ---- yaw damper + turn coordination -> rudder ----
        r_ref = p.g * np.sin(phi) / V            # yaw rate that belongs to a coordinated turn
        dr_cmd = self.c_trim[m6.DR] + s * (self.Kr * (r - r_ref) - self.Kb * beta)
        dr = np.clip(dr_cmd, -p.delta_r_max, p.delta_r_max)
        return da, dr


def bank_profile(t):
    """Bank command: 0° → 20° right at t = 5 s, back to 0° at t = 35 s."""
    return np.deg2rad(20.0) if 5.0 <= t < 35.0 else 0.0


def simulate(Kr=None, Kb=None, V0=50.0, h0=1000.0, t_end=60.0, dt=0.02):
    x, c0 = trim_6dof(V0, h=h0)
    lon = LongitudinalAutopilot(x, c0)           # holds altitude and speed
    lat = LateralAutopilot(x, c0)
    if Kr is not None:
        lat.Kr = Kr
    if Kb is not None:
        lat.Kb = Kb
    keys = ["t", "phi", "phi_set", "beta", "r", "psi", "h", "V", "da", "dr", "de", "T"]
    log = {k: [] for k in keys}
    for t in np.arange(0.0, t_end, dt):
        c, _ = lon.step(x, h0, V0, dt)
        c[m6.DA], c[m6.DR] = lat.step(x, bank_profile(t), dt)
        x = m6.rk4_step(x, c, dt)
        V, _, beta = m6.air_data(x)
        vals = [t, np.rad2deg(x[m6.PHI]), np.rad2deg(lat.phi_set), np.rad2deg(beta),
                np.rad2deg(x[m6.R]), np.rad2deg(x[m6.PSI]), x[m6.H], V,
                np.rad2deg(c[m6.DA]), np.rad2deg(c[m6.DR]), np.rad2deg(c[m6.DE]), c[m6.TH]]
        for k, v in zip(keys, vals):
            log[k].append(v)
    return {k: np.array(v) for k, v in log.items()}


def summary(r, h0=1000.0):
    turn = (r["t"] >= 15) & (r["t"] < 35)                     # steady part of the turn
    return {
        "max phi overshoot [deg]": max(0.0, r["phi"].max() - 20.0),
        "phi error in turn [deg]": np.abs(r["phi"] - r["phi_set"])[turn].max(),
        "max |beta| [deg]": np.abs(r["beta"]).max(),
        "max altitude loss [m]": h0 - r["h"].min(),
        "heading change [deg]": r["psi"][-1],
        "turn rate in turn [deg/s]": np.gradient(r["psi"], r["t"])[turn].mean(),
    }


if __name__ == "__main__":
    aircraft.load("c172")
    os.makedirs("results", exist_ok=True)
    runs = {"rudder fixed": simulate(Kr=0.0, Kb=0.0),
            "yaw damper (r only)": simulate(Kb=0.0),
            "yaw damper + beta feedback": simulate()}

    print(f"{'':28s}" + "".join(f"{n:>30s}" for n in runs))
    for k in summary(next(iter(runs.values()))):
        print(f"{k:28s}" + "".join(f"{summary(r)[k]:30.2f}" for r in runs.values()))
    print(f"(textbook turn rate for 20° at 50 m/s: {np.rad2deg(p.g * np.tan(np.deg2rad(20)) / 50):.2f} deg/s)")

    fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for (name, r), col in zip(runs.items(), ["#999999", "#8fb3ee", "#2a6fdb"]):
        ax[0, 0].plot(r["t"], r["phi"], color=col, label=name)
        ax[0, 1].plot(r["t"], r["beta"], color=col, label=name)
        ax[0, 2].plot(r["t"], r["h"] - 1000.0, color=col, label=name)
        ax[1, 0].plot(r["t"], r["da"], color=col, label=name)
        ax[1, 1].plot(r["t"], r["dr"], color=col, label=name)
        ax[1, 2].plot(r["t"], r["psi"], color=col, label=name)
    ax[0, 0].plot(r["t"], r["phi_set"], "k--", lw=1, label="φ_set (rate-limited)")
    titles = [["Bank angle φ [deg]", "Sideslip β [deg]", "Altitude deviation [m]"],
              ["Aileron δa [deg]", "Rudder δr [deg]", "Heading ψ [deg]"]]
    for i in range(2):
        for j in range(3):
            ax[i, j].set_title(titles[i][j]); ax[i, j].grid(); ax[i, j].legend(fontsize=8)
    for a in ax[1]:
        a.set_xlabel("Time [s]")
    fig.suptitle("C172 lateral PID autopilot: bank 20° right at t = 5 s, back to 0° at t = 35 s "
                 "(longitudinal autopilot holds h and V)")
    plt.tight_layout()
    plt.savefig("results/autopilot_lat_pid.png", dpi=120)
    plt.show()
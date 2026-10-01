"""
Heading hold: outer loop on top of the bank-angle autopilots (PID or LQI).

    psi_set ──▶ heading loop ──phi_cmd──▶ bank loop (PID or LQI) ──▶ aileron, rudder

Heading loop:
    e_psi       = wrap(psi_set - psi)                 shortest way round (-180° ... +180°)
    psi_dot_cmd = K_psi * e_psi                       desired turn rate
    phi_cmd     = arctan(V * psi_dot_cmd / g)         bank angle that gives this turn rate
                  limited to ±phi_max
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import aircraft
from aircraft import p
import flightdynamics_6dof as m6
from trim_6dof import trim_6dof
from autopilot_lon import LongitudinalAutopilot
from autopilot_lat import LateralAutopilot
from autopilot_lat_lqi import LQIAutopilotLat, build_schedule


def wrap(angle):
    """Map an angle to -pi ... +pi (e.g. 350° -> -10°)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class HeadingHold:
    K_psi = 0.3                                  # [1/s]: 10° heading error -> 3 °/s turn rate
    phi_max = np.deg2rad(25.0)                   # bank limit for heading changes

    def bank_command(self, x, psi_set):
        V = m6.air_data(x)[0]
        e_psi = wrap(psi_set - x[m6.PSI])                          # shortest way round
        psi_dot_cmd = self.K_psi * e_psi                           # desired turn rate
        phi_cmd = np.arctan(V * psi_dot_cmd / p.g)                 # bank for that turn rate
        return np.clip(phi_cmd, -self.phi_max, self.phi_max)


def heading_profile(t):
    """Heading command: 0° until t = 5 s, then 90° (right turn)."""
    return np.deg2rad(90.0) if t >= 5.0 else 0.0


def simulate(make_lat, V0=50.0, h0=1000.0, t_end=60.0, dt=0.02, psi_profile=heading_profile):
    x, c0 = trim_6dof(V0, h=h0)
    lon = LongitudinalAutopilot(x, c0)           # holds altitude and speed
    lat = make_lat(x, c0)                        # bank loop (PID or LQI)
    hdg = HeadingHold()
    keys = ["t", "psi", "psi_set", "phi", "beta", "h"]
    log = {k: [] for k in keys}
    for t in np.arange(0.0, t_end, dt):
        psi_set = psi_profile(t)
        c, _ = lon.step(x, h0, V0, dt)
        c[m6.DA], c[m6.DR] = lat.step(x, hdg.bank_command(x, psi_set), dt)
        x = m6.rk4_step(x, c, dt)
        _, _, beta = m6.air_data(x)
        vals = [t, np.rad2deg(x[m6.PSI]), np.rad2deg(psi_set), np.rad2deg(x[m6.PHI]),
                np.rad2deg(beta), x[m6.H]]
        for k, v in zip(keys, vals):
            log[k].append(v)
    return {k: np.array(v) for k, v in log.items()}


def summary(r, h0=1000.0):
    err = r["psi"] - r["psi_set"]
    after = r["t"] >= 5.0
    outside = np.where(np.abs(err[after]) > 2.0)[0]           # ±2° band
    return {
        "heading overshoot [deg]": max(0.0, (r["psi"] - 90.0).max()),
        "time to ±2° [s]": r["t"][after][outside[-1]] - 5.0 if len(outside) else 0.0,
        "max bank [deg]": np.abs(r["phi"]).max(),
        "max |beta| [deg]": np.abs(r["beta"]).max(),
        "max altitude loss [m]": h0 - r["h"].min(),
    }


if __name__ == "__main__":
    aircraft.load("c172")
    os.makedirs("results", exist_ok=True)
    schedule = build_schedule()
    pid = lambda x, c: LateralAutopilot(x, c)
    lqi = lambda x, c: LQIAutopilotLat(schedule, x)
    speeds = [30.0, 50.0, 65.0]
    runs = {(V, n): simulate(f, V0=V) for V in speeds for n, f in [("PID", pid), ("LQI", lqi)]}

    print(f"{'':26s}" + "".join(f"{f'{V:.0f} m/s {n}':>12s}" for V, n in runs))
    for k in summary(runs[(50.0, "PID")]):
        print(f"{k:26s}" + "".join(f"{summary(r)[k]:12.2f}" for r in runs.values()))

    # quick check of the wrap: from 350° to 10° the aircraft must turn RIGHT by 20°
    print("wrap check: wrap(10° - 350°) =", np.rad2deg(wrap(np.deg2rad(10 - 350))), "deg")

    fig, ax = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    for (V, n), r in runs.items():
        if V != 50.0:
            continue
        col = "#2a6fdb" if n == "PID" else "#e8782a"
        for a, key in zip(ax.flat, ["psi", "phi", "beta", "h"]):
            a.plot(r["t"], r[key] - (1000.0 if key == "h" else 0.0), color=col, label=n)
    ax[0, 0].plot(r["t"], r["psi_set"], "k--", lw=1, label="ψ_set")
    titles = ["Heading ψ [deg]", "Bank angle φ [deg]", "Sideslip β [deg]", "Altitude deviation [m]"]
    for a, tt in zip(ax.flat, titles):
        a.set_title(tt); a.grid(); a.legend(fontsize=8)
    for a in ax[1]:
        a.set_xlabel("Time [s]")
    fig.suptitle("C172 heading hold at 50 m/s: 90° heading change at t = 5 s – PID vs. LQI bank loop")
    plt.tight_layout()
    plt.savefig("results/autopilot_heading.png", dpi=120)
    plt.show()
"""
Phase 6, step 4: steep turns with altitude hold (longitudinal + lateral coupled).

Full PID system  = PID cascade (lon) + PID bank loop (lat)
Full LQI system  = LQI (lon)         + LQI bank loop (lat)
Each with and without the turn feedforward (turn_ff.py).

Scenario at 50 m/s: bank to phi at t = 5 s, hold, back to 0° at t = 35 s.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import aircraft
import flightdynamics_6dof as m6
from trim_6dof import trim_6dof
from autopilot_lon import LongitudinalAutopilot
from autopilot_lon_lqi import LQIAutopilotLon, build_schedule as build_lon
from autopilot_lat import LateralAutopilot
from autopilot_lat_lqi import LQIAutopilotLat, build_schedule as build_lat
from turn_ff import TurnFeedforward

V0, H0, DT, T_END = 50.0, 1000.0, 0.02, 60.0
BANKS = [30.0, 45.0, 60.0]


def simulate(system, ff, bank_deg, sched):
    x, c0 = trim_6dof(V0, h=H0)
    if system == "PID":
        lon, lat = LongitudinalAutopilot(x, c0), LateralAutopilot(x, c0)
    else:
        lon, lat = LQIAutopilotLon(sched["lon"]), LQIAutopilotLat(sched["lat"], x)
    lat.phi_max = np.deg2rad(65.0)               # allow steep turns in this test
    lon.turn_ff = sched["ff"] if ff else None
    keys = ["t", "phi", "h", "V", "de", "T"]
    log = {k: [] for k in keys}
    for t in np.arange(0.0, T_END, DT):
        phi_cmd = np.deg2rad(bank_deg) if 5.0 <= t < 35.0 else 0.0
        c, _ = lon.step(x, H0, V0, DT)
        c[m6.DA], c[m6.DR] = lat.step(x, phi_cmd, DT)
        x = m6.rk4_step(x, c, DT)
        vals = [t, np.rad2deg(x[m6.PHI]), x[m6.H], m6.air_data(x)[0],
                np.rad2deg(c[m6.DE]), c[m6.TH]]
        for k, v in zip(keys, vals):
            log[k].append(v)
    return {k: np.array(v) for k, v in log.items()}


def summary(r):
    return {"max |Δh| [m]": np.abs(r["h"] - H0).max(),
            "max |ΔV| [m/s]": np.abs(r["V"] - V0).max(),
            "max thrust [N]": r["T"].max()}


if __name__ == "__main__":
    aircraft.load("c172")
    os.makedirs("results", exist_ok=True)
    sched = {"lon": build_lon(), "lat": build_lat(), "ff": TurnFeedforward(V0)}
    variants = [("PID", False), ("PID", True), ("LQI", False), ("LQI", True)]
    runs = {(b, s, ff): simulate(s, ff, b, sched) for b in BANKS for s, ff in variants}

    names = [s + ("+FF" if ff else "") for s, ff in variants]
    for b in BANKS:
        print(f"\nBank {b:.0f}°" + "".join(f"{n:>10s}" for n in names))
        for k in summary(runs[(b, "PID", False)]):
            print(f"  {k:16s}" + "".join(f"{summary(runs[(b, s, ff)])[k]:10.1f}" for s, ff in variants))

    fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
    styles = {("PID", False): ("#2a6fdb", ":"), ("PID", True): ("#2a6fdb", "-"),
              ("LQI", False): ("#e8782a", ":"), ("LQI", True): ("#e8782a", "-")}
    for j, b in enumerate(BANKS):
        for (s, ff), (col, ls) in styles.items():
            r = runs[(b, s, ff)]
            lbl = s + ("+FF" if ff else "")
            ax[0, j].plot(r["t"], r["h"] - H0, ls, color=col, label=lbl)
            ax[1, j].plot(r["t"], r["V"] - V0, ls, color=col, label=lbl)
        ax[0, j].set_title(f"Bank {b:.0f}° – altitude deviation [m]")
        ax[1, j].set_title(f"Bank {b:.0f}° – speed deviation [m/s]")
        ax[1, j].set_xlabel("Time [s]")
    for a in ax.flat:
        a.grid(); a.legend(fontsize=8); a.axvspan(5, 35, color="#999999", alpha=0.1)
    fig.suptitle("C172 steep turns at 50 m/s with altitude hold: without (dotted) vs. with turn feedforward (solid)")
    plt.tight_layout()
    plt.savefig("results/turn_coupled.png", dpi=120)
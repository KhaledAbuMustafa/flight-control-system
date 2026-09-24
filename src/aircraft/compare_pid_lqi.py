"""
PID (2 separate loops) vs. LQI (one MIMO controller) on the nonlinear model.

Scenario 1: V +5 m/s and theta +3 deg commanded at the same time
Scenario 2: vertical 1-cos gust, controllers hold the trim point
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from trim import trim
from linearize import linearize
from simulation import simulate
from lqi import design_lqi, simulate_lqi

V0 = 50.0
dV, dtheta = 5.0, 3.0


def one_minus_cos_gust(w_max=5.0, t_start=2.0, duration=3.0):
    """Discrete vertical gust (as in CS-23): smooth bump from 0 to w_max and back."""
    def w(t):
        if t_start <= t <= t_start + duration:
            return 0.5 * w_max * (1 - np.cos(2 * np.pi * (t - t_start) / duration))
        return 0.0
    return w


def settling_time(t, y, target, band):
    outside = np.where(np.abs(y - target) > band)[0]
    return t[outside[-1]] if len(outside) else 0.0


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    Kx, Kz, _ = design_lqi(*linearize(*trim(V0)))

    # ---------------- Scenario 1: simultaneous steps ----------------
    pid = simulate(V0=V0, V_target=V0 + dV, theta_step_deg=dtheta, t_step=2.0, t_end=60.0)
    lqi = simulate_lqi(Kx, Kz, V0=V0, dV=dV, dtheta_deg=dtheta, t_step=2.0, t_end=60.0)

    print("Scenario 1: V +5 m/s and theta +3 deg at t = 2 s")
    for name, r in [("PID", pid), ("LQI", lqi)]:
        th_target = r["theta"][0] + dtheta
        print(f"  {name}: settling V (±0.25 m/s) = {settling_time(r['t'], r['V'], V0 + dV, 0.25) - 2:5.1f} s | "
              f"settling θ (±0.15°) = {settling_time(r['t'], r['theta'], th_target, 0.15) - 2:5.1f} s | "
              f"θ overshoot = {r['theta'].max() - th_target:4.2f}° | "
              f"δe range = {r['delta_e'].max() - r['delta_e'].min():4.1f}° | "
              f"T peak = {r['T'].max():5.0f} N")

    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for name, r in [("PID (2 loops)", pid), ("LQI (MIMO)", lqi)]:
        ax[0, 0].plot(r["t"], r["V"], label=name)
        ax[0, 1].plot(r["t"], r["theta"], label=name)
        ax[1, 0].plot(r["t"], r["delta_e"], label=name)
        ax[1, 1].plot(r["t"], r["T"], label=name)
    ax[0, 0].plot(lqi["t"], lqi["V_set"], "k--", label="setpoint")
    ax[0, 1].plot(lqi["t"], lqi["theta_set"], "k--", label="setpoint")
    ax[0, 0].set(ylabel="V [m/s]", title="Airspeed")
    ax[0, 1].set(ylabel="θ [deg]", title="Pitch angle")
    ax[1, 0].set(ylabel="δe [deg]", xlabel="Time [s]", title="Elevator")
    ax[1, 1].set(ylabel="T [N]", xlabel="Time [s]", title="Thrust")
    for a in ax.flat:
        a.grid(); a.legend()
    fig.suptitle("Scenario 1: V +5 m/s and θ +3° at the same time")
    plt.tight_layout()
    plt.savefig("results/compare_steps.png", dpi=120)

    # ---------------- Scenario 2: gust ----------------
    gust = one_minus_cos_gust()
    pid_g = simulate(V0=V0, V_target=V0, theta_step_deg=0.0, t_end=60.0, gust=gust)
    lqi_g = simulate_lqi(Kx, Kz, V0=V0, dV=0.0, dtheta_deg=0.0, t_end=60.0, gust=gust)

    print("\nScenario 2: 1-cos vertical gust, 5 m/s, 3 s")
    for name, r in [("PID", pid_g), ("LQI", lqi_g)]:
        print(f"  {name}: max |ΔV| = {np.abs(r['V'] - V0).max():4.2f} m/s | "
              f"max |Δθ| = {np.abs(r['theta'] - r['theta'][0]).max():4.2f}° | "
              f"max |Δγ| = {np.abs(r['gamma'] - r['gamma'][0]).max():4.2f}° | "
              f"recovered θ (±0.1°) after {settling_time(r['t'], r['theta'], r['theta'][0], 0.1) - 2:4.1f} s")

    t = pid_g["t"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for name, r in [("PID (2 loops)", pid_g), ("LQI (MIMO)", lqi_g)]:
        ax[0, 0].plot(r["t"], r["V"], label=name)
        ax[0, 1].plot(r["t"], r["theta"], label=name)
        ax[1, 0].plot(r["t"], r["gamma"], label=name)
        ax[1, 1].plot(r["t"], r["delta_e"], label=name)
    ax2 = ax[1, 0].twinx()
    ax2.fill_between(t, [gust(ti) for ti in t], color="grey", alpha=0.2)
    ax2.set_ylabel("gust w [m/s]", color="grey")
    ax[0, 0].set(ylabel="V [m/s]", title="Airspeed")
    ax[0, 1].set(ylabel="θ [deg]", title="Pitch angle")
    ax[1, 0].set(ylabel="γ [deg]", xlabel="Time [s]", title="Flight path angle (+ gust)")
    ax[1, 1].set(ylabel="δe [deg]", xlabel="Time [s]", title="Elevator")
    for a in ax.flat:
        a.grid(); a.legend()
    fig.suptitle("Scenario 2: vertical 1-cos gust (5 m/s, 3 s)")
    plt.tight_layout()
    plt.savefig("results/compare_gust.png", dpi=120)
    plt.show()
"""
Gain-scheduled LQI over airspeed.

Offline:  for each grid speed V_i -> trim (x0, u0) -> linearize (A, B) -> LQI (Kx, Kz)
Online:   gains      Kx(V), Kz(V)   interpolated at the MEASURED airspeed
          trim/ffwd  x0(V), u0(V)   interpolated at the COMMANDED airspeed
          u = u0(V_set) - Kx(V) (x - x0(V_set)) - Kz(V) z
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from flightdynamics import calculate_derivatives
from trim import trim
from linearize import linearize
from lqi import design_lqi, lqi_control

V_GRID = np.array([30.0, 40.0, 50.0, 60.0])


# ============================================================
# Offline: build the schedule table
# ============================================================
def build_schedule(V_grid=V_GRID):
    table = {"V": V_grid, "x0": [], "u0": [], "Kx": [], "Kz": []}
    for V in V_grid:
        x0, u0 = trim(V)
        A, B = linearize(x0, u0)
        Kx, Kz, _ = design_lqi(A, B)
        for key, val in zip(["x0", "u0", "Kx", "Kz"], [x0, u0, Kx, Kz]):
            table[key].append(val)
    for key in ["x0", "u0", "Kx", "Kz"]:
        table[key] = np.array(table[key])          # shape: (n_grid, ...)
    return table


def interpolate(table, key, V):
    """Linear interpolation of every entry of table[key] over V (clamped at the grid ends)."""
    data = table[key]
    flat = data.reshape(len(table["V"]), -1)
    values = [np.interp(V, table["V"], flat[:, i]) for i in range(flat.shape[1])]
    return np.array(values).reshape(data.shape[1:])


# ============================================================
# Online: closed-loop simulation
# ============================================================
def ramp(V_start, V_end, t_start, rate):
    """Commanded airspeed: constant, then a ramp with |dV/dt| = rate, then constant."""
    duration = abs(V_end - V_start) / rate
    def V_cmd(t):
        if t <= t_start:
            return V_start
        if t >= t_start + duration:
            return V_end
        return V_start + np.sign(V_end - V_start) * rate * (t - t_start)
    return V_cmd


def simulate(table, V_cmd, V_start, schedule_trim=True, schedule_gains=True,
             V_fixed=50.0, t_end=60.0, dt=0.01):
    """
    schedule_trim : x0, u0 follow the commanded speed (else fixed at V_fixed)
    schedule_gains: Kx, Kz follow the measured speed  (else fixed at V_fixed)
    Both False = one fixed LQI designed at V_fixed.
    All variants get the same reference: r = [V_set, theta_trim(V_set)]  (level flight).
    """
    x, u_start = trim(V_start)

    def operating_point(V_set, V_meas):
        x0 = interpolate(table, "x0", V_set if schedule_trim else V_fixed)
        u0 = interpolate(table, "u0", V_set if schedule_trim else V_fixed)
        Kx = interpolate(table, "Kx", V_meas if schedule_gains else V_fixed)
        Kz = interpolate(table, "Kz", V_meas if schedule_gains else V_fixed)
        return x0, u0, Kx, Kz

    # Bumpless start: choose z so that the controller initially outputs the trim input
    x0, u0, Kx, Kz = operating_point(V_start, V_start)
    z = np.linalg.solve(Kz, u0 - Kx @ (x - x0) - u_start)

    log = {k: [] for k in ["t", "V", "V_set", "theta", "theta_set", "delta_e", "T"]}

    for t in np.arange(0.0, t_end, dt):
        V_set = V_cmd(t)
        x0, u0, Kx, Kz = operating_point(V_set, x[0])

        theta_set = interpolate(table, "x0", V_set)[2]  # level flight attitude at V_set
        r = np.array([V_set, theta_set])

        u, z = lqi_control(x, x0, u0, z, r, Kx, Kz, dt)
        x = x + np.array(calculate_derivatives(*x, *u)) * dt

        for key, val in zip(log, [t, x[0], V_set, np.rad2deg(x[2]), np.rad2deg(theta_set),
                                  np.rad2deg(u[0]), u[1]]):
            log[key].append(val)

    return {k: np.array(v) for k, v in log.items()}


def tracking_errors(res, t_from=0.0):
    mask = res["t"] >= t_from
    return (np.abs(res["V"] - res["V_set"])[mask].max(),
            np.abs(res["theta"] - res["theta_set"])[mask].max())


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    os.makedirs("results", exist_ok=True)
    table = build_schedule()

    # ---------- How much do the gains actually change? ----------
    print("Elevator row of Kx  [V, gamma, theta, q]:")
    for V, Kx in zip(table["V"], table["Kx"]):
        print(f"  V = {V:.0f} m/s: {Kx[0]}")

    variants = {
        "fixed LQI (50 m/s)": dict(schedule_trim=False, schedule_gains=False),
        "only gains scheduled": dict(schedule_trim=False, schedule_gains=True),
        "only trim scheduled": dict(schedule_trim=True, schedule_gains=False),
        "fully scheduled": dict(schedule_trim=True, schedule_gains=True),
    }

    # ---------- Test 1: large transition 30 -> 60 m/s ----------
    V_cmd = ramp(30.0, 60.0, t_start=5.0, rate=1.0)
    print("\nTest 1: ramp 30 -> 60 m/s at 1 m/s^2")
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for name, opts in variants.items():
        r = simulate(table, V_cmd, 30.0, t_end=70.0, **opts)
        eV, eth = tracking_errors(r)
        print(f"  {name:22s}: max |V - V_set| = {eV:4.2f} m/s | max |θ - θ_set| = {eth:4.2f}°")
        ax[0, 0].plot(r["t"], r["V"] - r["V_set"], label=name)
        ax[0, 1].plot(r["t"], r["theta"] - r["theta_set"], label=name)
        ax[1, 0].plot(r["t"], r["delta_e"], label=name)
        ax[1, 1].plot(r["t"], r["T"], label=name)
    ax[0, 0].set(ylabel="V - V_set [m/s]", title="Speed tracking error")
    ax[0, 1].set(ylabel="θ - θ_set [deg]", title="Pitch tracking error")
    ax[1, 0].set(ylabel="δe [deg]", xlabel="Time [s]", title="Elevator")
    ax[1, 1].set(ylabel="T [N]", xlabel="Time [s]", title="Thrust")
    for a in ax.flat:
        a.grid(); a.legend(fontsize=8)
    fig.suptitle("Test 1: speed ramp 30 → 60 m/s (1 m/s²)")
    plt.tight_layout()
    plt.savefig("results/gs_ramp.png", dpi=120)

    # ---------- Test 2: small manoeuvres at the edges and between grid points ----------
    print("\nTest 2: 2 m/s speed change (ramped), fixed vs. fully scheduled")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for i, (V_a, V_b) in enumerate([(30.0, 32.0), (45.0, 47.0), (60.0, 58.0)]):
        V_cmd = ramp(V_a, V_b, t_start=2.0, rate=1.0)
        for name in ["fixed LQI (50 m/s)", "fully scheduled"]:
            r = simulate(table, V_cmd, V_a, t_end=30.0, **variants[name])
            eV, eth = tracking_errors(r)
            print(f"  {V_a:.0f} -> {V_b:.0f} m/s, {name:18s}: max |V err| = {eV:4.2f} m/s, max |θ err| = {eth:4.2f}°")
            ax[i].plot(r["t"], r["V"] - r["V_set"], label=name)
        ax[i].set(title=f"{V_a:.0f} → {V_b:.0f} m/s", xlabel="Time [s]")
        ax[i].grid(); ax[i].legend()
    ax[0].set_ylabel("V - V_set [m/s]")
    fig.suptitle("Test 2: small speed changes at the edges and between grid points")
    plt.tight_layout()
    plt.savefig("results/gs_steps.png", dpi=120)

    # ---------- Interpolation check: trim between grid points ----------
    print("\nInterpolated vs. true trim elevator:")
    for V in [35.0, 45.0, 55.0]:
        de_interp = np.rad2deg(interpolate(table, "u0", V)[0])
        de_true = np.rad2deg(trim(V)[1][0])
        print(f"  V = {V:.0f} m/s: interpolated {de_interp:5.2f}°, true {de_true:5.2f}°")
    plt.show()
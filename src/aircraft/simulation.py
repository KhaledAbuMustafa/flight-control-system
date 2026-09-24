import os
import numpy as np
import matplotlib.pyplot as plt
import c172_params as p
from flightdynamics import calculate_derivatives
from trim import trim
from controller import pitch_controller, speed_pi_controller


# ============================================================
# Controller gains
# ============================================================
# Pitch PID (holds the trim pitch attitude)
Kp, Kd, Ki = 1.5, 0.2, 0.6          # designed in pitch_design.py

# Speed PI: plant approx. V_dot = T/m  ->  choose closed-loop wn, zeta
wn_v, zeta_v = 0.5, 0.8
Kp_v = 2 * zeta_v * wn_v * p.m            # [N per m/s]
Ki_v = wn_v**2 * p.m                      # [N per m]


def simulate(V0=50.0, V_target=60.0, theta_step_deg=0.0, t_step=5.0, t_end=120.0, dt=0.01,
             anti_windup=True, pitch_gains=None, gust=None):

    Kp_th, Kd_th, Ki_th = pitch_gains if pitch_gains is not None else (Kp, Kd, Ki)

    # Start exactly in trim
    x0, u0 = trim(V0)
    V, gamma, theta, q = x0
    delta_e_trim, T_trim = u0

    theta_trim = theta                    # trim attitude = starting point
    I_theta, I_v = 0.0, 0.0
    elevator_saturated = False

    log = {k: [] for k in ["t", "V", "V_set", "T", "gamma", "theta", "theta_set", "delta_e"]}

    for t in np.arange(0.0, t_end, dt):

        V_set = V_target if t >= t_step else V0
        theta_set = theta_trim + (np.deg2rad(theta_step_deg) if t >= t_step else 0.0)

        # Pitch PID with anti-windup
        if not elevator_saturated:
            I_theta += (theta_set - theta) * dt
        delta_e_cmd = delta_e_trim + pitch_controller(theta_set, theta, q, I_theta, Kp_th, Kd_th, Ki_th)
        delta_e = np.clip(delta_e_cmd, -p.delta_e_max, p.delta_e_max)
        elevator_saturated = (delta_e != delta_e_cmd)

        # Speed PI with anti-windup
        T, I_v = speed_pi_controller(V_set, V, I_v, Kp_v, Ki_v, T_trim,
                                     p.T_min, p.T_max, dt, anti_windup)

        # Aircraft model + Euler integration
        V_dot, gamma_dot, theta_dot, q_dot = calculate_derivatives(
            V, gamma, theta, q, delta_e, T, gust(t) if gust else 0.0)
        V += V_dot * dt
        gamma += gamma_dot * dt
        theta += theta_dot * dt
        q += q_dot * dt

        for key, val in zip(log, [t, V, V_set, T, np.rad2deg(gamma), np.rad2deg(theta), np.rad2deg(theta_set), np.rad2deg(delta_e)]):
            log[key].append(val)

    return {k: np.array(v) for k, v in log.items()}


if __name__ == "__main__":

    with_aw = simulate(anti_windup=True)
    without_aw = simulate(anti_windup=False)

    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    for res, label in [(without_aw, "without anti-windup"), (with_aw, "with anti-windup")]:
        ax[0, 0].plot(res["t"], res["V"], label=label)
        ax[0, 1].plot(res["t"], res["T"], label=label)
        ax[1, 0].plot(res["t"], res["gamma"], label=label)
        ax[1, 1].plot(res["t"], res["theta"], label=label)

    ax[0, 0].plot(with_aw["t"], with_aw["V_set"], "k--", label="V_set")
    ax[0, 1].axhline(p.T_max, color="k", linestyle="--", label="T_max")

    ax[0, 0].set_ylabel("Airspeed V [m/s]")
    ax[0, 1].set_ylabel("Thrust T [N]")
    ax[1, 0].set_ylabel("Flight path γ [deg]")
    ax[1, 1].set_ylabel("Pitch θ [deg]")
    ax[1, 0].set_xlabel("Time [s]")
    ax[1, 1].set_xlabel("Time [s]")

    for a in ax.flat:
        a.grid()
        a.legend()

    fig.suptitle(f"Speed step 50 → 60 m/s  (Kp_v = {Kp_v:.0f}, Ki_v = {Ki_v:.0f})")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/speed_pi_antiwindup.png", dpi=120)
    plt.show()
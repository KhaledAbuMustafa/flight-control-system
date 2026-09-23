"""
Pitch PID design in three steps:
    A) Kd  - pitch damper: improve short-period damping
    B) Kp  - attitude loop: sets the speed of the response
    C) Ki  - removes the steady-state error
Then: check on the nonlinear model at several airspeeds.
"""
import os
import numpy as np
import control as ct
import matplotlib.pyplot as plt
from trim import trim
from linearize import linearize
from simulation import simulate

OLD = (3.0, 0.3, 0.5)          # (Kp, Kd, Ki) old, hand-tuned
NEW = (1.5, 0.2, 0.6)          # designed below


def closed_loop(A, B, Kp, Kd, Ki):
    """
    Linear closed loop: states [dV, dgamma, dtheta, q, z], z = integral of pitch error.
    Input: theta_set.  Outputs: [theta, delta_e].
    Control law: delta_e = -Kp*(theta_set - theta) - Ki*z + Kd*q
    """
    b = B[:, 0]                                  # elevator column of B
    Acl = np.zeros((5, 5))
    Acl[:4, :4] = A                              # aircraft
    Acl[:4, 2] += b * Kp                         # +Kp*theta  (from -Kp*(theta_set - theta))
    Acl[:4, 3] += b * Kd                         # +Kd*q
    Acl[:4, 4] += -b * Ki                        # -Ki*z
    Acl[4, 2] = -1.0                             # z_dot = theta_set - theta
    Bcl = np.zeros((5, 1))
    Bcl[:4, 0] = -b * Kp                         # -Kp*theta_set
    Bcl[4, 0] = 1.0
    C = np.array([[0, 0, 1, 0, 0],               # output 1: theta
                  [0, 0, Kp, Kd, -Ki]])          # output 2: delta_e
    D = np.array([[0.0], [-Kp]])
    return ct.ss(Acl, Bcl, C, D)


def step_metrics(t, y):
    """Rise time 10-90 %, overshoot, 5 % settling time for a unit step."""
    i10, i90 = np.argmax(y >= 0.1), np.argmax(y >= 0.9)
    rise = t[i90] - t[i10] if y.max() >= 0.9 else np.nan
    overshoot = max(0.0, (y.max() - 1.0) * 100)
    outside = np.where(np.abs(y - 1.0) > 0.05)[0]
    settling = t[outside[-1]] if len(outside) else 0.0
    return rise, overshoot, settling


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    A, B = linearize(*trim(50.0))
    b = B[:, 0]

    # ---------- Step A: pitch damper (Kd only) ----------
    print("Step A - short-period mode vs. Kd (V = 50 m/s)")
    for Kd in [0.0, 0.1, 0.2, 0.3]:
        Ak = A.copy()
        Ak[:, 3] += b * Kd                        # close only the q-loop
        ev = np.linalg.eigvals(Ak)
        sp = max(ev, key=abs)                     # fastest eigenvalue = short period
        print(f"  Kd = {Kd:.1f}: wn = {abs(sp):.2f} rad/s, zeta = {-sp.real/abs(sp):.2f}")

    # ---------- Steps B + C: linear step responses ----------
    t = np.linspace(0, 20, 4001)
    cases = {"P + D  (Ki = 0)": (NEW[0], NEW[1], 0.0), "PID new": NEW, "PID old": OLD}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    print("\nSteps B/C - linear model, unit step on theta_set (V = 50 m/s)")
    for name, K in cases.items():
        _, y = ct.step_response(closed_loop(A, B, *K), t)
        th, de = y[0, 0], y[1, 0]
        r, o, s = step_metrics(t, th)
        print(f"  {name:16s}: rise = {r:.2f} s, overshoot = {o:4.1f} %, "
              f"settling = {s:4.1f} s, final = {th[-1]:.2f}, elevator per 5 deg = {abs(de).max()*5:.1f} deg")
        ax[0].plot(t, th, label=name)
        ax[1].plot(t, de * 5, label=name)          # unit step response scaled to a 5 deg step
    ax[0].axhline(1, color="k", ls="--")
    ax[0].set(title="Linear: θ / θ_set", xlabel="Time [s]", ylabel="[-]")
    ax[1].set(title="Linear: elevator for a 5° step", xlabel="Time [s]", ylabel="δe [deg]")
    for a in ax:
        a.grid(); a.legend()
    plt.tight_layout()
    plt.savefig("results/pitch_linear_steps.png", dpi=120)

    # ---------- Check on the nonlinear model ----------
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for V in [30.0, 50.0, 60.0]:
        r = simulate(V0=V, V_target=V, theta_step_deg=5.0, t_step=2.0, t_end=30.0, pitch_gains=NEW)
        ax[0, 0].plot(r["t"], r["theta"] - r["theta"][0], label=f"V = {V:.0f} m/s")
        ax[1, 0].plot(r["t"], r["delta_e"], label=f"V = {V:.0f} m/s")
    ax[0, 0].axhline(5, color="k", ls="--")
    ax[0, 0].set(title="Nonlinear, new gains: 5° step", ylabel="Δθ [deg]")
    ax[1, 0].set(ylabel="δe [deg]", xlabel="Time [s]")

    for K, name in [(OLD, "old gains"), (NEW, "new gains")]:
        r = simulate(V0=50.0, V_target=50.0, theta_step_deg=10.0, t_step=2.0, t_end=30.0, pitch_gains=K)
        ax[0, 1].plot(r["t"], r["theta"] - r["theta"][0], label=name)
        ax[1, 1].plot(r["t"], r["delta_e"], label=name)
    ax[0, 1].axhline(10, color="k", ls="--")
    ax[1, 1].axhline(-25, color="r", ls=":", label="elevator limit")
    ax[0, 1].set(title="Nonlinear, V = 50 m/s: 10° step", ylabel="Δθ [deg]")
    ax[1, 1].set(ylabel="δe [deg]", xlabel="Time [s]")
    for a in ax.flat:
        a.grid(); a.legend()
    plt.tight_layout()
    plt.savefig("results/pitch_nonlinear_check.png", dpi=120)
    plt.show()
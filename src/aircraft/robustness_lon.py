"""
Step 1c: robustness of the longitudinal autopilots (PID cascade vs. gain-scheduled LQI).

Conditions:  aircraft (C172, C182) x airspeed (30 / 50 / 65 m/s) x CG (forward / reference / aft)
Tests:       A) altitude hold in a vertical 1-cos gust (5 m/s, 3 s)
             B) climb of +50 m with a 5 m/s ramp
Important:   the LQI schedule is designed at the REFERENCE CG only. A shifted CG is a model
             error the controller does not know about -> this is a real robustness test.
"""
import os
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
import aircraft
from aircraft import p
import flightdynamics_6dof as m6
from trim_6dof import trim_6dof
from autopilot_lon import LongitudinalAutopilot
from autopilot_lon_lqi import build_schedule, LQIAutopilotLon, interp, LON, INPUTS

SPEEDS = [30.0, 50.0, 65.0]
CGS = {"fwd": -0.25, "ref": 0.0, "aft": 0.10}
DT, T_END = 0.02, 40.0          # RK4 is accurate enough with 20 ms steps


def gust(t, w_max=5.0, t0=5.0, dur=3.0):
    return 0.5 * w_max * (1 - np.cos(2 * np.pi * (t - t0) / dur)) if t0 <= t <= t0 + dur else 0.0


def make_lqi(schedule, x, c_trim):
    """LQI with bumpless start: integrators chosen so that the first command equals the true trim."""
    ctrl = LQIAutopilotLon(schedule)
    V = m6.air_data(x)[0]
    x0 = interp(schedule, "x0", V); x0[4] = x[m6.H]
    c0 = interp(schedule, "c0", V)
    Kx, Kz = interp(schedule, "Kx", V), interp(schedule, "Kz", V)
    ctrl.z = np.linalg.solve(Kz, c0[INPUTS] - Kx @ (x[LON] - x0) - c_trim[INPUTS])
    return ctrl


def run(controller, test, V0, h0=1000.0):
    x, c_trim = trim_6dof(V0, h=h0)                 # trimmed at the ACTUAL CG (the real aircraft)
    ctrl = controller(x, c_trim)
    h_err, V_err, diverged = [], [], False
    for t in np.arange(0.0, T_END, DT):
        h_set = h0 + (np.clip((t - 5.0) * 5.0, 0.0, 50.0) if test == "climb" else 0.0)
        wind = (0.0, 0.0, -gust(t)) if test == "gust" else (0.0, 0.0, 0.0)
        c, _ = ctrl.step(x, h_set, V0, DT)
        x = m6.rk4_step(x, c, DT, wind)
        h_err.append(x[m6.H] - h_set)
        V_err.append(m6.air_data(x)[0] - V0)
        if abs(h_err[-1]) > 500 or not np.all(np.isfinite(x)):
            diverged = True
            break
    return {"h": np.max(np.abs(h_err)), "V": np.max(np.abs(V_err)), "diverged": diverged}


_SCHEDULES = {}                     # per-process cache: one LQI schedule per aircraft


def task(args):
    """One simulation. Runs in a separate process -> everything it needs is passed in."""
    name, cg_name, dh, V, test, ctrl = args
    aircraft.load(name)
    if name not in _SCHEDULES:
        p.dh_cg = 0.0
        _SCHEDULES[name] = build_schedule()      # designed at the reference CG
    schedule = _SCHEDULES[name]
    p.dh_cg = dh                                 # the "real" aircraft has a shifted CG
    if ctrl == "PID":
        factory = lambda x, c: LongitudinalAutopilot(x, c)
    else:
        factory = lambda x, c: make_lqi(schedule, x, c)
    result = run(factory, test, V)
    p.dh_cg = 0.0
    return (name, cg_name, V, test, ctrl), result


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    jobs = [(name, cg_name, dh, V, test, ctrl)
            for name in ["c172", "c182"] for cg_name, dh in CGS.items()
            for V in SPEEDS for test in ["gust", "climb"] for ctrl in ["PID", "LQI"]]
    t0 = time.time()
    with Pool(cpu_count()) as pool:                   # one worker per CPU core
        results = dict(pool.map(task, jobs))
    print(f"{len(jobs)} simulations on {cpu_count()} cores in {time.time() - t0:.1f} s")

    # ---------- table ----------
    for test, label in [("gust", "A) gust: max |Δh| [m] / max |ΔV| [m/s]"),
                        ("climb", "B) climb +50 m: max |h error| [m] / max |ΔV| [m/s]")]:
        print(f"\n{label}")
        print(f"{'':14s}" + "".join(f"{f'{V:.0f} m/s':>26s}" for V in SPEEDS))
        print(f"{'':14s}" + "".join(f"{'PID':>13s}{'LQI':>13s}" for _ in SPEEDS))
        for name in ["c172", "c182"]:
            for cg_name in CGS:
                row = f"{name.upper()} {cg_name:4s}     "
                for V in SPEEDS:
                    for ctrl in ["PID", "LQI"]:
                        r = results[(name, cg_name, V, test, ctrl)]
                        row += f"{'DIVERGED':>13s}" if r["diverged"] else f"{r['h']:7.1f}/{r['V']:4.2f}"
                print(row)

    # ---------- figure: grouped bars ----------
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharey="col")
    cond = [(cg, V) for cg in CGS for V in SPEEDS]
    xs = np.arange(len(cond))
    for i, name in enumerate(["c172", "c182"]):
        for j, (test, ylabel) in enumerate([("gust", "gust: max |Δh| [m]"),
                                            ("climb", "climb: max |h error| [m]")]):
            ax = axes[i, j]
            for k, (ctrl, col) in enumerate([("PID", "#2a6fdb"), ("LQI", "#e8782a")]):
                vals = [results[(name, cg, V, test, ctrl)]["h"] for cg, V in cond]
                ax.bar(xs + (k - 0.5) * 0.38, vals, width=0.36, color=col, label=ctrl)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{cg}\n{V:.0f}" for cg, V in cond], fontsize=8)
            ax.set_title(f"{name.upper()} – {ylabel}")
            ax.grid(axis="y")
            ax.legend()
    for ax in axes[1]:
        ax.set_xlabel("CG position / airspeed [m/s]")
    fig.suptitle("Robustness: PID (fixed gains) vs. LQI (scheduled over V, designed at reference CG)")
    plt.tight_layout()
    plt.savefig("results/robustness_lon.png", dpi=120)
    plt.show()
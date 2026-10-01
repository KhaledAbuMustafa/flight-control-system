"""
Phase 6 summary: PID vs. LQI on all autopilot tasks (C172, 50 m/s unless stated).

Both controllers always get the same references and the same feedforward,
so the comparison shows the controller structure, not the tuning tricks.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import aircraft
import autopilot_lon_lqi as lon_lqi
import robustness_lon as rob
import autopilot_lat_lqi as lat_lqi
import autopilot_heading as hdg
import turn_coupled as tc
from autopilot_lat import LateralAutopilot, summary as lat_summary
from turn_ff import TurnFeedforward

PID_COL, LQI_COL = "#2a6fdb", "#e8782a"


def collect(name="c172"):
    aircraft.load(name)
    lon_s, lat_s = lon_lqi.build_schedule(), lat_lqi.build_schedule()
    pid_lat = lambda x, c: LateralAutopilot(x, c)
    lqi_lat = lambda x, c: lat_lqi.LQIAutopilotLat(lat_s, x)
    res = {}

    # 1) climb +100 m with climb-rate feedforward
    r = {"PID": lon_lqi.simulate(name, lon_lqi.pid_factory_ff(True)),
         "LQI": lon_lqi.simulate(name, lon_lqi.lqi_factory_for(name, True))}
    res["Climb +100 m\nmax altitude error [m]"] = {
        k: lon_lqi.summary(v)["max |h err| climb [m]"] for k, v in r.items()}

    # 2) vertical gust 5 m/s
    aircraft.load(name)                          # simulate() above resets to the default aircraft
    g = {"PID": rob.run(lambda x, c: rob.LongitudinalAutopilot(x, c), "gust", 50.0),
         "LQI": rob.run(lambda x, c: rob.make_lqi(lon_s, x, c), "gust", 50.0)}
    res["Vertical gust 5 m/s\nmax altitude deviation [m]"] = {k: v["h"] for k, v in g.items()}
    res["Vertical gust 5 m/s\nmax speed deviation [m/s]"] = {k: v["V"] for k, v in g.items()}

    # 3) bank 20° at low speed (fixed PID gains are designed for 50 m/s)
    b = {"PID": lat_lqi.simulate(pid_lat, V0=30.0), "LQI": lat_lqi.simulate(lqi_lat, V0=30.0)}
    res["Bank 20° at 30 m/s\nmax sideslip [deg]"] = {
        k: lat_summary(v)["max |beta| [deg]"] for k, v in b.items()}

    # 4) heading change 90°
    h = {"PID": hdg.simulate(pid_lat), "LQI": hdg.simulate(lqi_lat)}
    res["Heading change 90°\ntime to ±2° [s]"] = {
        k: hdg.summary(v)["time to ±2° [s]"] for k, v in h.items()}

    # 5) steep turn 45° with turn feedforward
    sched = {"lon": lon_s, "lat": lat_s, "ff": TurnFeedforward(50.0)}
    t = {k: tc.simulate(k, True, 45.0, sched) for k in ["PID", "LQI"]}
    res["Steep turn 45°\nmax altitude deviation [m]"] = {
        k: tc.summary(v)["max |Δh| [m]"] for k, v in t.items()}
    return res


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "c172"      # e.g.  python compare_all.py c182
    os.makedirs("results", exist_ok=True)
    res = collect(name)

    print(f"{'task':48s}{'PID':>9s}{'LQI':>9s}   better")
    for task, v in res.items():
        better = "LQI" if v["LQI"] < v["PID"] else "PID"
        print(f"{task.replace(chr(10), ' – '):48s}{v['PID']:9.2f}{v['LQI']:9.2f}   {better}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (task, v) in zip(axes.flat, res.items()):
        bars = ax.bar(["PID", "LQI"], [v["PID"], v["LQI"]], color=[PID_COL, LQI_COL], width=0.55)
        for bar, val in zip(bars, [v["PID"], v["LQI"]]):
            ax.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
        ax.set_title(task, fontsize=10)
        ax.set_ylim(0, max(v.values()) * 1.25)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{name.upper()} autopilot: PID cascade vs. LQI, both gain-scheduled over airspeed (lower is better)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"results/compare_all_{name}.png", dpi=120)
    plt.show()
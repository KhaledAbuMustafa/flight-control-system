"""
Compare the flying-qualities maps of two aircraft (same model, same criteria, same grid).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import aircraft
from aircraft import p
from trim_6dof import trim_6dof, linearize_6dof
from modes import identify_modes
from level_map import level_grid, CMAP, NORM, LEVEL_COLORS

AIRCRAFT = ["c172", "c182"]
V_values = np.arange(30.0, 65.01, 2.5)
dh_values = np.round(np.arange(-0.50, 0.201, 0.05), 3)


def summary(name):
    aircraft.load(name)
    grids = level_grid(V_values, dh_values, "B")
    ok = [dh for i, dh in enumerate(dh_values) if np.all(grids["overall"][i] <= 2)]
    A, _ = linearize_6dof(*trim_6dof(50.0))
    modes = identify_modes(A, 50.0)
    return {"name": p.NAME, "grids": grids, "ok": ok, "c_bar": p.c_bar,
            "dh_np": -p.Cm_alpha / p.CL_alpha, "modes": modes, "m": p.m,
            "x_ref": getattr(p, "x_cg_ref_mac", None)}


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    results = [summary(a) for a in AIRCRAFT]

    print(f"{'':32s}" + "".join(f"{r['name']:>30s}" for r in results))
    rows = [
        ("mass [kg]", lambda r: f"{r['m']:.0f}"),
        ("static margin at ref. CG [% c̄]", lambda r: f"{r['dh_np'] * 100:.1f}"),
        ("CG window width [cm]", lambda r: f"{(max(r['ok']) - min(r['ok'])) * r['c_bar'] * 100:.0f}" if r["ok"] else "-"),
        ("Level 1 points [%]", lambda r: f"{np.mean(r['grids']['overall'] == 1) * 100:.0f}"),
        ("short period @50 (wn/zeta)", lambda r: f"{r['modes']['short_period']['wn']:.2f} / {r['modes']['short_period']['zeta']:.2f}"),
        ("phugoid @50 (wn/zeta)", lambda r: f"{r['modes']['phugoid']['wn']:.3f} / {r['modes']['phugoid']['zeta']:.3f}"),
        ("dutch roll @50 (wn/zeta)", lambda r: f"{r['modes']['dutch_roll']['wn']:.2f} / {r['modes']['dutch_roll']['zeta']:.2f}"),
        ("roll mode @50 tau [s]", lambda r: f"{r['modes']['roll']['tau']:.3f}"),
        ("spiral @50 tau [s]", lambda r: f"{r['modes']['spiral'].get('tau', np.nan):.0f}"),
    ]
    for label, fn in rows:
        print(f"{label:32s}" + "".join(f"{fn(r):>30s}" for r in results))
    for r in results:
        if r["x_ref"] is not None and r["ok"]:
            print(f"\n{r['name']}: CG window in % c̄ = {(r['x_ref'] + min(r['ok'])) * 100:.0f} % "
                  f"... {(r['x_ref'] + max(r['ok'])) * 100:.0f} %, neutral point ≈ "
                  f"{(r['x_ref'] + r['dh_np']) * 100:.0f} % c̄")

    # ---------- side-by-side overall maps ----------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, r in zip(axes, results):
        g = r["grids"]["overall"]
        ax.pcolormesh(V_values, dh_values, g, cmap=CMAP, norm=NORM, shading="nearest",
                      edgecolors="white", linewidth=1.5)
        for i, dh in enumerate(dh_values):
            for j, V in enumerate(V_values):
                ax.text(V, dh, str(g[i, j]), ha="center", va="center", fontsize=7, color="#1a1a19")
        ax.axhline(r["dh_np"], color="#1a1a19", ls="--", lw=1.2)
        if r["ok"]:
            step = dh_values[1] - dh_values[0]
            for y in (min(r["ok"]) - step / 2, max(r["ok"]) + step / 2):
                ax.axhline(y, color="#1a1a19", lw=3)
            width = (max(r["ok"]) - min(r["ok"])) * r["c_bar"] * 100
            ax.set_title(f"{r['name']}\nCG window ≈ {width:.0f} cm, "
                         f"static margin {r['dh_np'] * 100:.0f} % c̄")
        ax.set_xlabel("Airspeed V [m/s]")
    axes[0].set_ylabel("CG shift from the data's reference CG [c̄]   (aft ↑)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LEVEL_COLORS]
    fig.legend(handles, ["Level 1", "Level 2", "Level 3", "not acceptable"], loc="lower center",
               ncol=4, frameon=False)
    fig.suptitle("Overall flying qualities, open loop (MIL-F-8785C Class I, Cat. B)   "
                 "— dashed: neutral point, thick: allowed CG window", fontsize=12)
    plt.tight_layout(rect=(0, 0.06, 1, 0.95))
    plt.savefig("results/compare_c172_c182.png", dpi=120)
    aircraft.load(aircraft.DEFAULT)
    plt.show()
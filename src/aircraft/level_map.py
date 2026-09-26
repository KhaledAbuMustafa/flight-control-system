"""
Flying-qualities level map: airspeed x CG position.

For every grid point: set CG -> trim -> linearize -> identify modes -> evaluate levels.
Output: one map per eigenmode + an overall map (worst level), coloured like a traffic light.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import c172_params as p
from trim_6dof import trim_6dof, linearize_6dof
from modes import identify_modes
from criteria import evaluate

MODES = ["short_period", "phugoid", "roll", "dutch_roll", "spiral"]
CHECKS = MODES + ["elevator"]
TITLES = {"short_period": "Short period", "phugoid": "Phugoid", "roll": "Roll mode",
          "dutch_roll": "Dutch roll", "spiral": "Spiral",
          "elevator": "Elevator authority (trim)", "overall": "OVERALL (worst)"}

# Own design assumption (not from MIL-F-8785C): keep at least this much elevator
# travel in reserve for manoeuvring and gusts beyond the trim deflection.
ELEVATOR_RESERVE = np.deg2rad(5.0)


def level_elevator(delta_e_trim):
    """1 = enough reserve, 3 = trimmable but reserve < 5°, 4 = beyond the stop."""
    lo, hi = p.delta_e_min, p.delta_e_max_6dof
    if delta_e_trim < lo or delta_e_trim > hi:
        return 4
    if delta_e_trim < lo + ELEVATOR_RESERVE or delta_e_trim > hi - ELEVATOR_RESERVE:
        return 3
    return 1

# Status colours: Level 1 good, 2 warning, 3 serious, 4 critical (not acceptable)
LEVEL_COLORS = ["#0ca30c", "#fab219", "#ec835a", "#d03b3b"]
CMAP = ListedColormap(LEVEL_COLORS)
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], CMAP.N)


def level_grid(V_values, dh_values, category="B"):
    """Return {mode: 2D array of levels} with shape (len(dh), len(V))."""
    grids = {m: np.full((len(dh_values), len(V_values)), 4) for m in CHECKS + ["overall"]}
    for i, dh in enumerate(dh_values):
        for j, V in enumerate(V_values):
            p.dh_cg = dh
            try:
                x0, c0 = trim_6dof(V)
                A, _ = linearize_6dof(x0, c0)
                levels, overall = evaluate(identify_modes(A, V), category)
                levels["elevator"] = level_elevator(c0[0])
                overall = max(overall, levels["elevator"])
            except (RuntimeError, KeyError, ValueError):
                continue                             # no trim / mode missing -> stays Level 4
            finally:
                p.dh_cg = 0.0
            for m in CHECKS:
                grids[m][i, j] = levels[m]
            grids["overall"][i, j] = overall
    return grids


def plot_maps(grids, V_values, dh_values, category, filename):
    dh_np = -p.Cm_alpha / p.CL_alpha
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), sharex=True, sharey=True)
    for ax, key in zip(axes.flat, CHECKS + ["overall"]):
        g = grids[key]
        ax.pcolormesh(V_values, dh_values, g, cmap=CMAP, norm=NORM,
                      shading="nearest", edgecolors="white", linewidth=1.5)
        for i, dh in enumerate(dh_values):              # level number in every cell (not colour alone)
            for j, V in enumerate(V_values):
                ax.text(V, dh, str(g[i, j]), ha="center", va="center", fontsize=7, color="#1a1a19")
        ax.axhline(dh_np, color="#1a1a19", ls="--", lw=1.2)
        ax.set_title(TITLES[key], fontweight="bold" if key == "overall" else "normal")

    # Allowed CG window = rows where every speed is Level <= 2 overall
    ok_rows = [dh for i, dh in enumerate(dh_values) if np.all(grids["overall"][i] <= 2)]
    ax_all, ax_win = axes.flat[6], axes.flat[7]
    step = dh_values[1] - dh_values[0]
    if ok_rows:
        lo, hi = min(ok_rows) - step / 2, max(ok_rows) + step / 2
        for y in (lo, hi):
            ax_all.axhline(y, color="#1a1a19", lw=3)

        # Summary panel: the CG window as one bar
        ax_win.axhspan(dh_values[0] - step / 2, lo, color=LEVEL_COLORS[3], alpha=0.85)
        ax_win.axhspan(lo, hi, color=LEVEL_COLORS[0], alpha=0.85)
        ax_win.axhspan(hi, dh_values[-1] + step / 2, color=LEVEL_COLORS[3], alpha=0.85)
        xm = (V_values[0] + V_values[-1]) / 2
        ax_win.text(xm, (lo + hi) / 2, f"ALLOWED CG WINDOW\n≈ {(max(ok_rows) - min(ok_rows)) * p.c_bar * 100:.0f} cm",
                    ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        ax_win.text(xm, (dh_values[0] - step / 2 + lo) / 2, "too nose-heavy\n(elevator stop)",
                    ha="center", va="center", fontsize=10, color="white")
        ax_win.text(xm, (hi + dh_values[-1] + step / 2) / 2, "unstable",
                    ha="center", va="center", fontsize=10, color="white")
        ax_win.axhline(0, color="white", ls=":", lw=1.5)
        ax_win.text(V_values[-1], 0, "reference CG ", ha="right", va="bottom", fontsize=8, color="white")
        ax_win.set_title("Result: CG window (Level ≤ 2 at all speeds)", fontweight="bold")
        ax_win.tick_params(labelbottom=False, bottom=False)
    fig.text(0.5, 0.075, "dashed line = neutral point (static stability limit)", ha="center",
             fontsize=9, color="#555555")
    for ax in axes[1]:
        ax.set_xlabel("Airspeed V [m/s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("CG shift dh [c̄]   (aft ↑)")


    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LEVEL_COLORS]
    labels = ["Level 1 – good", "Level 2 – acceptable", "Level 3 – controllable", "not acceptable"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"C172 flying qualities (MIL-F-8785C, Class I, Category {category}), open loop",
                 fontsize=14)
    plt.tight_layout(rect=(0, 0.09, 1, 0.96))
    plt.savefig(filename, dpi=120)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    V_values = np.arange(30.0, 60.01, 2.5)
    dh_values = np.round(np.arange(-0.50, 0.201, 0.05), 3)

    for category in ["B", "C"]:
        grids = level_grid(V_values, dh_values, category)
        plot_maps(grids, V_values, dh_values, category, f"results/level_map_cat{category}.png")
        n = grids["overall"].size
        print(f"Category {category}: " + ", ".join(
            f"Level {k}: {np.sum(grids['overall'] == k)}/{n}" for k in (1, 2, 3, 4)))
        ok = [dh for i, dh in enumerate(dh_values) if np.all(grids["overall"][i] <= 2)]
        if ok:
            print(f"  allowed CG window (Level <= 2 over {V_values[0]:.0f}-{V_values[-1]:.0f} m/s): "
                  f"dh = {min(ok):+.2f} ... {max(ok):+.2f} c_bar  "
                  f"(= {(max(ok) - min(ok)) * p.c_bar * 100:.0f} cm)")
    plt.show()
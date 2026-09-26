"""
Effect of the CG position on stability and trim (6-DOF model, straight and level flight).

    dh_cg = CG position aft of the data reference point, as a fraction of c_bar
    Neutral point (estimate):  dh_np = -Cm_alpha / CL_alpha
    Static margin:             SM = dh_np - dh_cg
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import c172_params as p
import flightdynamics_6dof as m6
from trim_6dof import trim_6dof, linearize_6dof

LON = [m6.U, m6.W, m6.Q, m6.THETA]
LAT = [m6.V_, m6.P, m6.R, m6.PHI]


def eigen_at(V, dh):
    """Trim + linearize at airspeed V with CG shift dh; return lon/lat eigenvalues and trim."""
    p.dh_cg = dh
    try:
        x0, c0 = trim_6dof(V)
        A, _ = linearize_6dof(x0, c0)
    finally:
        p.dh_cg = 0.0                                # always restore the reference CG
    return (np.linalg.eigvals(A[np.ix_(LON, LON)]),
            np.linalg.eigvals(A[np.ix_(LAT, LAT)]), x0, c0)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    dh_np = -p.Cm_alpha / p.CL_alpha
    print(f"Neutral point (estimate): dh_np = {dh_np:.3f} c_bar = {dh_np * p.c_bar * 100:.0f} cm aft of reference")

    V = 50.0
    dh_values = np.linspace(-0.15, 0.22, 38)

    # ---------- longitudinal eigenvalues vs. CG ----------
    rows = []
    for dh in dh_values:
        lon, lat, x0, c0 = eigen_at(V, dh)
        rows.append((dh, lon, lat, c0[0]))

    print("\n dh_cg   SM     oscillatory modes, slow -> fast (wn / zeta)     max Re(lon)")
    for dh, lon, lat, de in rows[::4]:
        cplx = sorted([e for e in lon if e.imag > 1e-6], key=abs)
        txt = "  ".join(f"{abs(e):5.2f} / {-e.real/abs(e):5.2f}" for e in cplx)
        print(f" {dh:+.3f}  {dh_np - dh:+.3f}   {txt:40s}  {lon.real.max():+.3f}")

    # Numerical stability boundary: first dh where an eigenvalue has a positive real part
    unstable = [dh for dh, lon, _, _ in rows if lon.real.max() > 0]
    if unstable:
        print(f"\nFirst unstable CG in the sweep: dh = {unstable[0]:+.3f} c_bar "
              f"(analytic neutral point {dh_np:.3f})")

    # ---------- forward CG limit from elevator authority at low speed ----------
    print("\nTrim elevator at 30 m/s (limit: -28° = full nose up):")
    for dh in [-0.20, -0.10, 0.0, 0.10]:
        _, _, _, c0 = eigen_at(30.0, dh)
        print(f"  dh = {dh:+.2f}: delta_e = {np.rad2deg(c0[0]):6.2f}°")
    # linear in dh -> where does trim reach the elevator stop?
    de_a = eigen_at(30.0, 0.0)[3][0]
    de_b = eigen_at(30.0, -0.1)[3][0]
    slope = (de_a - de_b) / 0.1
    dh_fwd = (p.delta_e_min - de_a) / slope
    print(f"  -> elevator stop reached at dh = {dh_fwd:+.2f} c_bar (forward CG limit, 30 m/s, no margin)")

    # ---------- plots ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    cmap = plt.cm.viridis
    for i, (dh, lon, lat, de) in enumerate(rows):
        col = cmap(i / (len(rows) - 1))
        ax[0].scatter(lon.real, lon.imag, color=col, s=12)
    ax[0].axvline(0, color="r", lw=1)
    ax[0].set(title=f"Longitudinal eigenvalues vs. CG (V = {V:.0f} m/s)",
              xlabel="Re [1/s]", ylabel="Im [rad/s]")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(dh_values[0], dh_values[-1]))
    fig.colorbar(sm, ax=ax[0], label="dh_cg [c_bar]  (aft →)")

    dhs = np.array([r[0] for r in rows])
    zeta_sp = []
    for _, lon, _, _ in rows:
        c = [e for e in lon if e.imag > 1e-6 and abs(e) > 1.0]
        zeta_sp.append(-c[0].real / abs(c[0]) if c else np.nan)
    ax[1].plot(dhs, [max(r[1].real) for r in rows], label="largest real part")
    ax[1].axhline(0, color="r", lw=1)
    ax[1].axvline(dh_np, color="k", ls="--", label="neutral point")
    ax[1].set(title="Stability margin vs. CG", xlabel="dh_cg [c_bar]", ylabel="max Re(λ) [1/s]")
    ax[1].legend()

    ax[2].plot(dhs, [np.rad2deg(r[3]) for r in rows])
    ax[2].axvline(dh_np, color="k", ls="--", label="neutral point")
    ax[2].set(title=f"Trim elevator vs. CG (V = {V:.0f} m/s)", xlabel="dh_cg [c_bar]", ylabel="δe [deg]")
    ax[2].legend()
    for a in ax:
        a.grid()
    plt.tight_layout()
    plt.savefig("results/cg_analysis.png", dpi=120)
    plt.show()
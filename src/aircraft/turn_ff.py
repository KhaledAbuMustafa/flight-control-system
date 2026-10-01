"""
Turn feedforward for the longitudinal autopilots.

In a level turn at bank angle phi the aircraft needs (all taken from the 6-DOF turn trim):
    more lift  -> higher alpha / pitch attitude, more up-elevator
    more drag  -> more thrust
    a pitch rate q = psi_dot*sin(phi)*cos(theta) -> the controller must NOT damp this q away

The table is built once per airspeed by trimming coordinated turns over a bank grid,
then interpolated at the MEASURED bank angle (the lift need depends on the actual bank).
"""
import numpy as np
from aircraft import p
import flightdynamics_6dof as m6
from trim_6dof import trim_6dof

PHI_GRID = np.deg2rad(np.arange(0.0, 60.01, 5.0))
ZERO = {"du": 0.0, "dw": 0.0, "q": 0.0, "dtheta": 0.0, "dde": 0.0, "dT": 0.0}


class TurnFeedforward:
    def __init__(self, V, h=1000.0):
        x0, c0 = trim_6dof(V, h=h)                                   # straight flight
        rows = []
        for phi in PHI_GRID:
            x, c = trim_6dof(V, psi_dot=p.g * np.tan(phi) / V, h=h)  # coordinated level turn
            rows.append([x[m6.U] - x0[m6.U], x[m6.W] - x0[m6.W], x[m6.Q],
                         x[m6.THETA] - x0[m6.THETA], c[m6.DE] - c0[m6.DE], c[m6.TH] - c0[m6.TH]])
        self.table = np.array(rows)                                  # one row per bank angle

    def __call__(self, phi):
        a = min(abs(phi), PHI_GRID[-1])          # left and right turns need the same
        vals = [np.interp(a, PHI_GRID, self.table[:, i]) for i in range(6)]
        return dict(zip(ZERO, vals))
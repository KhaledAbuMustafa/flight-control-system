import numpy as np


def pitch_controller(theta_set, theta, q, I_err, Kp, Kd, Ki):
    """
    PID controller for pitch angle (elevator).

    Sign convention: Cm_delta_e < 0, i.e. positive delta_e = nose down.

    Returns:
        delta_e : elevator correction around trim [rad]
    """
    error = theta_set - theta

    # P: nose too low (error > 0) -> negative delta_e -> nose up
    # I: removes the steady-state error
    # D: positive q (nose rising) -> positive delta_e -> damping
    return -Kp * error - Ki * I_err + Kd * q


def speed_pi_controller(V_set, V, I_v, Kp_v, Ki_v, T_trim, T_min, T_max, dt, anti_windup=True):
    """
    PI controller for airspeed (thrust) with anti-windup (conditional integration).

    Returns:
        T   : saturated thrust command [N]
        I_v : updated integrator state [m]
    """
    error = V_set - V
    T_cmd = T_trim + Kp_v * error + Ki_v * I_v
    T = np.clip(T_cmd, T_min, T_max)

    saturated = (T != T_cmd)
    pushing_further = np.sign(error) == np.sign(T_cmd - T)

    if not (anti_windup and saturated and pushing_further):
        I_v += error * dt

    return T, I_v
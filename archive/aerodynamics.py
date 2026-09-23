import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Aircraft parameters
# ============================================================

S = 14.0              # Wing area [m²]
rho = 1.225           # Air density [kg/m³]
m = 800.0             # Aircraft mass [kg]
I_y = 1000.0          # Pitch moment of inertia [kg·m²]
g = 9.81              # Gravitational acceleration [m/s²]

# Aerodynamic coefficients
CL0 = 0.30
CL_alpha = 5.0        # Lift curve slope [1/rad]

CD0 = 0.030
k = 0.0558

# Engine
T = 3000.0            # Thrust [N]

# Pitching moment
c_bar = 1.4           # Mean aerodynamic chord [m]
Cm0 = 0.05
Cm_alpha = -1.0       # Pitch moment coefficient per rad
Cm_delta_e = -1.0     # Pitch moment coefficient per elevator deflection


# ============================================================
# Aerodynamics
# ============================================================

def lift_coefficient(alpha):
    """
    Calculate lift coefficient CL.

    alpha: angle of attack [rad]
    """

    CL = CL0 + CL_alpha * alpha

    return CL


def drag_coefficient(CL):
    """
    Calculate drag coefficient CD.

    CL: lift coefficient
    """

    CD = CD0 + k * CL**2

    return CD


def calculate_lift(V, CL):
    """
    Calculate lift force.

    V: airspeed [m/s]
    CL: lift coefficient
    """

    L = 0.5 * rho * V**2 * S * CL

    return L


def calculate_drag(V, CD):
    """
    Calculate drag force.

    V: airspeed [m/s]
    CD: drag coefficient
    """

    D = 0.5 * rho * V**2 * S * CD

    return D


# ============================================================
# Flight dynamics - translation
# ============================================================

def calculate_acceleration(T, D):
    """
    Calculate longitudinal acceleration.

    T: thrust [N]
    D: drag [N]
    """

    acceleration = (T - D) / m

    return acceleration


# ============================================================
# Flight dynamics - pitch
# ============================================================

def calculate_pitch_acceleration(M):
    """
    Calculate angular acceleration around the pitch axis.

    M: pitching moment [Nm]

    Returns:
        pitch_acceleration: angular acceleration [rad/s²]
    """

    pitch_acceleration = M / I_y

    return pitch_acceleration


# ============================================================
# Pitching moment
# ============================================================

def pitching_moment_coefficient(alpha, delta_e):
    """
    Calculate pitching moment coefficient.

    alpha: angle of attack [rad]
    delta_e: elevator deflection [rad]
    """

    Cm = Cm0 + Cm_alpha * alpha + Cm_delta_e * delta_e

    return Cm


def calculate_pitching_moment(V, Cm):
    """
    Calculate pitching moment.

    V: airspeed [m/s]
    Cm: pitching moment coefficient
    """

    M = 0.5 * rho * V**2 * S * c_bar * Cm

    return M


# ============================================================
# Trim and equilibrium
# ============================================================

def find_equilibrium_speed(alpha_deg, T):
    """
    Find the speed where thrust approximately equals drag.

    alpha_deg: angle of attack [deg]
    T: thrust [N]
    """

    alpha_rad = np.deg2rad(alpha_deg)

    CL = lift_coefficient(alpha_rad)
    CD = drag_coefficient(CL)

    speeds = np.linspace(1.0, 100.0, 1000)

    drag = 0.5 * rho * speeds**2 * S * CD

    difference = np.abs(T - drag)

    index = np.argmin(difference)

    equilibrium_speed = speeds[index]

    return equilibrium_speed


def find_trim_condition(T):
    """
    Search for a flight condition where:

        Lift  ≈ Weight
        Drag  ≈ Thrust

    T: thrust [N]
    """

    weight = m * g

    best_error = float("inf")
    best_speed = None
    best_alpha = None

    speeds = np.linspace(20.0, 100.0, 400)
    alphas = np.linspace(0.0, 10.0, 240)

    for V in speeds:

        for alpha_deg in alphas:

            alpha_rad = np.deg2rad(alpha_deg)

            CL = lift_coefficient(alpha_rad)
            CD = drag_coefficient(CL)

            L = calculate_lift(V, CL)
            D = calculate_drag(V, CD)

            lift_error = L - weight
            drag_error = D - T

            error = lift_error**2 + drag_error**2

            if error < best_error:

                best_error = error
                best_speed = V
                best_alpha = alpha_deg

    return best_speed, best_alpha


def find_true_trim():
    """
    Find a simplified trim point:

        M = 0
        L = W

    The required thrust is then calculated.
    """

    weight = m * g

    # M = 0:
    # Cm0 + Cm_alpha * alpha = 0
    alpha = -Cm0 / Cm_alpha

    # CL at this alpha
    CL = lift_coefficient(alpha)

    # Velocity from L = W
    V = np.sqrt(weight / (0.5 * rho * S * CL))

    # Drag at this velocity
    CD = drag_coefficient(CL)
    D = calculate_drag(V, CD)

    return V, np.rad2deg(alpha), D


# ============================================================
# Pitch simulation - original model
# ============================================================

def simulate_longitudinal():
    """
    Simulate simplified longitudinal aircraft dynamics.

    States:
        theta: pitch angle [rad]
        gamma: flight path angle [rad]
        q: pitch rate [rad/s]
    """

    # Initial conditions
    theta = np.deg2rad(5.0)
    gamma = np.deg2rad(0.0)
    q = 0.0

    V = 30.0

    # Simulation settings
    dt = 0.01
    simulation_time = 10.0

    time = np.arange(0.0, simulation_time, dt)

    pitch_angle = []
    flight_path_angle = []
    alpha_history = []

    # No elevator input
    delta_e = 0.0

    # Simulation loop
    for t in time:

        # Angle of attack
        alpha = theta - gamma

        # Aerodynamic coefficients
        CL = lift_coefficient(alpha)
        CD = drag_coefficient(CL)

        # Aerodynamic forces
        L = calculate_lift(V, CL)
        D = calculate_drag(V, CD)

        # Pitching moment
        Cm = pitching_moment_coefficient(alpha, delta_e)
        M = calculate_pitching_moment(V, Cm)

        # Accelerations
        acceleration = calculate_acceleration(T, D)
        pitch_acceleration = calculate_pitch_acceleration(M)

        # Flight path angle change
        gamma_dot = (L - m * g) / (m * V)

        # Update states
        q = q + pitch_acceleration * dt
        theta = theta + q * dt
        gamma = gamma + gamma_dot * dt
        V = V + acceleration * dt

        # Store results
        pitch_angle.append(np.rad2deg(theta))
        flight_path_angle.append(np.rad2deg(gamma))
        alpha_history.append(np.rad2deg(alpha))

    return time, pitch_angle, flight_path_angle, alpha_history


# ============================================================
# Speed simulation
# ============================================================

def simulate_speed():
    """
    Simulate the aircraft's longitudinal acceleration.

    Returns:
        time: simulation time
        velocity: airspeed [m/s]
    """

    alpha_deg = 5.0
    alpha_rad = np.deg2rad(alpha_deg)

    V = 30.0

    dt = 0.1
    simulation_time = 10.0

    time = np.arange(0.0, simulation_time, dt)

    velocity = []

    for t in time:

        CL = lift_coefficient(alpha_rad)
        CD = drag_coefficient(CL)

        L = calculate_lift(V, CL)
        D = calculate_drag(V, CD)

        acceleration = calculate_acceleration(T, D)

        V = V + acceleration * dt

        velocity.append(V)

    return time, velocity


# ============================================================
# Simulation from trim
# ============================================================

def simulate_from_trim():
    V, alpha_trim_deg, T_trim = find_true_trim()

    theta = np.deg2rad(alpha_trim_deg)
    gamma = 0.0
    q = 0.0

    dt = 0.01
    simulation_time = 10.0

    time = np.arange(0.0, simulation_time, dt)

    pitch_angle = []
    flight_path_angle = []
    alpha_history = []
    velocity_history = []

    # Kleine Störung
    theta += np.deg2rad(1.0)

    delta_e = 0.0

    for t in time:

        # Angle of attack
        alpha = theta - gamma

        # Aerodynamics
        CL = lift_coefficient(alpha)
        CD = drag_coefficient(CL)

        L = calculate_lift(V, CL)
        D = calculate_drag(V, CD)

        # Pitching moment
        Cm = pitching_moment_coefficient(alpha, delta_e)
        M = calculate_pitching_moment(V, Cm)

        # Longitudinal dynamics
        pitch_acceleration = calculate_pitch_acceleration(M)

        V_dot = (
            (T_trim - D) / m
            - g * np.sin(gamma)
        )

        gamma_dot = (
            L - m * g * np.cos(gamma)
        ) / (m * V)

        # Integration
        q = q + pitch_acceleration * dt
        theta = theta + q * dt

        gamma = gamma + gamma_dot * dt
        V = V + V_dot * dt

        # Save values
        pitch_angle.append(np.rad2deg(theta))
        flight_path_angle.append(np.rad2deg(gamma))
        alpha_history.append(np.rad2deg(alpha))
        velocity_history.append(V)

    return (
        time,
        pitch_angle,
        flight_path_angle,
        alpha_history,
        velocity_history
    )
# ============================================================
# Simulation with elevator input
# ============================================================

def simulate_elevator_input():

    V, alpha_trim_deg, T_trim = find_true_trim()

    theta = np.deg2rad(alpha_trim_deg)
    gamma = 0.0
    q = 0.0

    # Constant elevator deflection
    delta_e = np.deg2rad(2.0)

    dt = 0.01
    simulation_time = 10.0

    time = np.arange(0.0, simulation_time, dt)

    pitch_angle = []
    flight_path_angle = []
    alpha_history = []

    for t in time:

        # Angle of attack
        alpha = theta - gamma

        # Aerodynamics
        CL = lift_coefficient(alpha)
        CD = drag_coefficient(CL)

        # Aerodynamic forces
        L = calculate_lift(V, CL)
        D = calculate_drag(V, CD)

        # Pitching moment including elevator
        Cm = pitching_moment_coefficient(alpha, delta_e)
        M = calculate_pitching_moment(V, Cm)

        # Accelerations
        acceleration = calculate_acceleration(T_trim, D)
        pitch_acceleration = calculate_pitch_acceleration(M)

        # Flight path angle
        gamma_dot = (L - m * g) / (m * V)

        # Update states
        q = q + pitch_acceleration * dt
        theta = theta + q * dt

        gamma = gamma + gamma_dot * dt
        V = V + acceleration * dt

        # Store results
        pitch_angle.append(np.rad2deg(theta))
        flight_path_angle.append(np.rad2deg(gamma))
        alpha_history.append(np.rad2deg(alpha))

    return time, pitch_angle, flight_path_angle, alpha_history


# ============================================================
# Main program
# ============================================================

# ------------------------------------------------------------
# Speed simulation
# ------------------------------------------------------------

time_speed, velocity = simulate_speed()

plt.figure()
plt.plot(time_speed, velocity)

plt.xlabel("Time [s]")
plt.ylabel("Airspeed [m/s]")
plt.title("Aircraft acceleration")
plt.grid()

plt.show()


# ------------------------------------------------------------
# Equilibrium speed
# ------------------------------------------------------------

equilibrium_speed = find_equilibrium_speed(5.0, T)

print("----- EQUILIBRIUM SPEED -----")
print("Equilibrium speed:", equilibrium_speed, "m/s")


# ------------------------------------------------------------
# Lift vs. weight at equilibrium speed
# ------------------------------------------------------------

alpha_deg = 5.0
alpha_rad = np.deg2rad(alpha_deg)

CL = lift_coefficient(alpha_rad)

lift = calculate_lift(equilibrium_speed, CL)
weight = m * g

print("Lift:", lift, "N")
print("Weight:", weight, "N")
print("Lift - Weight:", lift - weight)


# ------------------------------------------------------------
# Trim condition
# ------------------------------------------------------------

trim_speed, trim_alpha = find_trim_condition(T)

print()
print("----- TRIM CONDITION -----")
print("Trim speed:", trim_speed, "m/s")
print("Trim alpha:", trim_alpha, "deg")


# ------------------------------------------------------------
# Longitudinal simulation
# ------------------------------------------------------------

time, pitch_angle, gamma_history, alpha_history = simulate_longitudinal()

plt.figure()

plt.plot(time, pitch_angle, label="Pitch angle θ")
plt.plot(time, gamma_history, label="Flight path angle γ")
plt.plot(time, alpha_history, label="Angle of attack α")

plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("Longitudinal aircraft dynamics")

plt.legend()
plt.grid()

plt.show()


# ------------------------------------------------------------
# Response around trim
# ------------------------------------------------------------

# ------------------------------------------------------------
# Simulation around trim
# ------------------------------------------------------------

time, pitch_angle, flight_path_angle, alpha_history, velocity_history = simulate_from_trim()

V_trim, alpha_trim_deg, T_trim = find_true_trim()

print()
print("=== Simplified trim point ===")
print(f"Trim speed:      {V_trim:.2f} m/s")
print(f"Trim alpha:      {alpha_trim_deg:.2f} deg")
print(f"Required thrust: {T_trim:.2f} N")


# ------------------------------------------------------------
# Angles around trim
# ------------------------------------------------------------

plt.figure()

plt.plot(time, pitch_angle, label="Pitch angle θ")
plt.plot(time, flight_path_angle, label="Flight path angle γ")
plt.plot(time, alpha_history, label="Angle of attack α")

plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("Aircraft response around trim")

plt.legend()
plt.grid()

plt.show()


# ------------------------------------------------------------
# Velocity around trim
# ------------------------------------------------------------

plt.figure()

plt.plot(time, velocity_history)

plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Velocity around trim")

plt.grid()

plt.show()


# ------------------------------------------------------------
# Elevator input
# ------------------------------------------------------------

time, pitch_angle, flight_path_angle, alpha_history = simulate_elevator_input()

plt.figure()

plt.plot(time, pitch_angle, label="Pitch angle θ")
plt.plot(time, flight_path_angle, label="Flight path angle γ")
plt.plot(time, alpha_history, label="Angle of attack α")

plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("Response to elevator input")

plt.legend()
plt.grid()

plt.show()
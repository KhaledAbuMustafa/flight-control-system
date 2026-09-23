import numpy as np
from flightdynamics import calculate_derivatives
from trim import trim


def f(x, u):
    """Nonlinear model as vector function: x_dot = f(x, u)."""
    return np.array(calculate_derivatives(*x, *u))


def linearize(x0, u0):
    """
    Linearize x_dot = f(x, u) around (x0, u0) with central differences.

    Returns:
        A : (4x4)  d f / d x
        B : (4x2)  d f / d u
    """
    n = len(x0)
    m = len(u0)
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = 1e-6 * max(1.0, abs(x0[i]))
        A[:, i] = (f(x0 + dx, u0) - f(x0 - dx, u0)) / (2 * dx[i])

    for j in range(m):
        du = np.zeros(m)
        du[j] = 1e-6 * max(1.0, abs(u0[j]))
        B[:, j] = (f(x0, u0 + du) - f(x0, u0 - du)) / (2 * du[j])

    return A, B


def modes(A):
    """Print eigenvalues with natural frequency and damping ratio."""
    for lam in np.linalg.eigvals(A):
        if lam.imag < 0:
            continue                                   # show each complex pair once
        wn = abs(lam)
        zeta = -lam.real / wn
        period = 2 * np.pi / lam.imag if lam.imag > 0 else np.inf
        print(f"  lambda = {lam.real:8.4f} {lam.imag:+8.4f}j | "
              f"wn = {wn:6.3f} rad/s | zeta = {zeta:5.3f} | period = {period:6.1f} s")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    V = 50.0
    x0, u0 = trim(V)
    A, B = linearize(x0, u0)

    print(f"Operating point V = {V} m/s")
    print("x = [V, gamma, theta, q],  u = [delta_e, T]\n")
    print("A =\n", A)
    print("\nB =\n", B)
    print("\nEigenmodes:")
    modes(A)
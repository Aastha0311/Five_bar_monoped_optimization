import numpy as np
from qpsolvers import solve_qp as qp
from scipy.sparse import csc_matrix
from scipy.linalg import expm

GRAVITY = 9.81


# ======================================================
# Reference trajectory generation (centroidal)
# ======================================================
def generate_reference_trajectory(initial_state,
                                  desired_velocity,
                                  desired_ang_vel,
                                  time_horizon,
                                  dt,
                                  desired_height):
    """
    Generates a COM reference trajectory for MPC.

    State: [x, y, theta, xdot, ydot, thetadot, gravity]
    """

    N = time_horizon
    n_states = len(initial_state)
    trajectory = np.zeros((N, n_states))

    for i in range(N):
        t = i * dt
        trajectory[i, 0] = initial_state[0] + desired_velocity[0] * t
        trajectory[i, 1] = desired_height
        trajectory[i, 2] = initial_state[2] + desired_ang_vel * t
        trajectory[i, 3] = desired_velocity[0]
        trajectory[i, 4] = 0.0
        trajectory[i, 5] = desired_ang_vel
        trajectory[i, 6] = -GRAVITY

    return trajectory


# ======================================================
# Lifted dynamics matrices (exact discretization)
# ======================================================
def get_lifted_dynamics_matrices(A, B, time_horizon, dt):
    """
    Builds lifted system:
        X = A_lifted x0 + B_lifted U
    """

    n_states = A.shape[0]
    n_inputs = B.shape[1]
    N = time_horizon

    # Exact discretization
    M = np.zeros((n_states + n_inputs, n_states + n_inputs))
    M[:n_states, :n_states] = A * dt
    M[:n_states, n_states:] = B * dt

    Md = expm(M)
    Ad = Md[:n_states, :n_states]
    Bd = Md[:n_states, n_states:]

    # Lifted matrices
    A_lifted = np.zeros((n_states * N, n_states))
    B_lifted = np.zeros((n_states * N, n_inputs * N))

    A_power = Ad.copy()
    for i in range(N):
        A_lifted[i*n_states:(i+1)*n_states, :] = A_power
        A_power = A_power @ Ad

    A_power = np.eye(n_states)
    for i in range(N):
        A_i_B = A_power @ Bd
        B_lifted[i*n_states:(i+1)*n_states, 0:n_inputs] = A_i_B

        if i > 0:
            B_lifted[i*n_states:(i+1)*n_states, n_inputs:(i+1)*n_inputs] = \
                B_lifted[(i-1)*n_states:i*n_states, 0:i*n_inputs]

        A_power = A_power @ Ad

    return A_lifted, B_lifted


# ======================================================
# Force constraints (single-foot contact)
# ======================================================
def get_force_constraints(friction_coeff, mass, time_horizon):
    """
    Single-foot force constraints per time step:
      Fy >= 0
      Fy <= 1.5 m g
      |Fx| <= mu Fy
    """

    g = GRAVITY
    mu = friction_coeff
    max_Fy = 1.5 * mass * g

    n_forces = 2        # Fx, Fy
    N = time_horizon
    n_constraints = 4 * N

    C = np.zeros((n_constraints, n_forces * N))
    c = np.zeros(n_constraints)

    row = 0
    for t in range(N):
        col = t * n_forces
        fx = col
        fy = col + 1

        # Fy >= 0  → -Fy <= 0
        C[row, fy] = -1
        c[row] = 0
        row += 1

        # Fy <= max_Fy
        C[row, fy] = 1
        c[row] = max_Fy
        row += 1

        # Fx >= -mu Fy  → -Fx - mu Fy <= 0
        C[row, fx] = -1
        C[row, fy] = -mu
        c[row] = 0
        row += 1

        # Fx <= mu Fy → Fx - mu Fy <= 0
        C[row, fx] = 1
        C[row, fy] = -mu
        c[row] = 0
        row += 1

    return c, C


# ======================================================
# QP solver
# ======================================================
def solve_qp(x_init,
             x_ref,
             A_lifted,
             B_lifted,
             force_constraints,
             Q_state=None,
             R_input=None):
    """
    Solves convex MPC QP.
    """

    n_states = A_lifted.shape[1]
    n_states_total = A_lifted.shape[0]
    n_inputs_total = B_lifted.shape[1]
    N = n_states_total // n_states
    n_inputs = n_inputs_total // N

    # ----------------------------
    # Cost weights
    # ----------------------------
    if Q_state is None:
        Q_diag = np.zeros(n_states)
        Q_diag[0] = 2.0    # x
        Q_diag[1] = 20.0   # y
        Q_diag[2] = 10.0   # theta
        Q_diag[3] = 5.0    # xdot
        Q_diag[4] = 5.0    # ydot
        Q_diag[5] = 5.0    # thetadot
        Q_diag[6] = 0.0    # gravity
        Q_state = np.diag(Q_diag)

    if R_input is None:
        R_input = np.eye(n_inputs) * 1e-4

    decay = 0.99
    L = np.block([
        [decay**i * Q_state if i == j else np.zeros((n_states, n_states))
         for j in range(N)]
        for i in range(N)
    ])

    K = np.block([
        [decay**i * R_input if i == j else np.zeros((n_inputs, n_inputs))
         for j in range(N)]
        for i in range(N)
    ])

    # ----------------------------
    # QP matrices
    # ----------------------------
    H = 2 * (B_lifted.T @ L @ B_lifted + K)
    g = 2 * B_lifted.T @ L @ (A_lifted @ x_init - x_ref.flatten())

    H += 1e-5 * np.eye(H.shape[0])  # regularization

    c, C = force_constraints

    H_sparse = csc_matrix(H)
    C_sparse = csc_matrix(C)

    # ----------------------------
    # Solve
    # ----------------------------
    try:
        res = qp(
            H_sparse, g, C_sparse, c,
            solver="osqp",
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=10000,
            polish=True,
            verbose=False
        )
        if res is not None:
            return res
    except Exception:
        pass

    try:
        res = qp(H_sparse, g, C_sparse, c, solver="clarabel", verbose=False)
        if res is not None:
            return res
    except Exception:
        pass

    print("WARNING: MPC QP failed — returning zero forces")
    return np.zeros(n_inputs_total)

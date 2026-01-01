import numpy as np
from qpsolvers import solve_qp as qp

GRAVITY = 9.81  # Avoid circular import

def generate_reference_trajectory(initial_state, desired_velocity, desired_ang_vel, time_horizon, dt, desired_height):
    """
    Generates a reference trajectory for the biped robot.

    Parameters:
    - initial_state: np.array, initial state of the robot [x, y, theta, x_dot, y_dot, theta_dot, gravity]
    - desired_velocity: np.array, desired horizontal velocity [vx, vy]
    - desired_ang_vel: float, desired angular velocity
    - time_horizon: int, number of time steps in the horizon
    - dt: float, time step
    - desired_height: float, desired COM height above ground

    Returns:
    - trajectory: np.array of shape (N, 7), where N is the number of time steps
    """
    N = time_horizon
    trajectory = np.zeros((N, len(initial_state)))

    for i in range(N):
        t = i * dt
        trajectory[i, 0] = initial_state[0] + desired_velocity[0] * t  # x position
        trajectory[i, 1] = desired_height  # y position - fixed desired height!
        trajectory[i, 2] = initial_state[2] + desired_ang_vel * t  # theta
        trajectory[i, 3] = desired_velocity[0]                         # x velocity
        trajectory[i, 4] = 0.0  # y velocity - should be zero at desired height
        trajectory[i, 5] = desired_ang_vel                          # angular velocity
        trajectory[i, 6] = -GRAVITY                           # gravity

    return trajectory


def get_lifted_dynamics_matrices(A, B, time_horizon, dt):
    """
    Computes the lifted dynamics matrices for MPC efficiently by reusing matrix multiplications.
    Discretizes the continuous-time system first using matrix exponential.
    Note: x_simulated_traj = A_lifted * x0 + B_lifted * u_sequence

    Parameters:
    - A: np.array, continuous-time state transition matrix
    - B: np.array, continuous-time control input matrix
    - time_horizon: int, number of time steps in the horizon
    - dt: float, time step

    Returns:
    - A_lifted: np.array, lifted state transition matrix
    - B_lifted: np.array, lifted control input matrix
    """
    from scipy.linalg import expm

    n_states = A.shape[0]
    n_inputs = B.shape[1]
    N = time_horizon

    # Discretize the continuous-time system using matrix exponential
    # Create augmented matrix [A B; 0 0]
    M = np.zeros((n_states + n_inputs, n_states + n_inputs))
    M[:n_states, :n_states] = A * dt
    M[:n_states, n_states:] = B * dt

    # Compute matrix exponential
    Md = expm(M)

    # Extract discretized matrices
    Ad = Md[:n_states, :n_states]
    Bd = Md[:n_states, n_states:]

    # Now build lifted matrices using discretized system
    A_lifted = np.zeros((n_states * N, n_states))
    B_lifted = np.zeros((n_states * N, n_inputs * N))

    # Incrementally build A_lifted by multiplying by Ad at each step
    A_power = Ad.copy()  # Ad^1
    for i in range(N):
        A_lifted[i*n_states:(i+1)*n_states, :] = A_power
        if i < N - 1:
            A_power = A_power @ Ad  # Ad^(i+2) = Ad^(i+1) @ Ad

    # Build B_lifted efficiently by reusing previous rows
    A_power = np.eye(n_states)  # Start with Ad^0 = I
    for i in range(N):
        # Compute Ad^i @ Bd for the leftmost block
        A_i_B = A_power @ Bd

        # Set the leftmost block
        B_lifted[i*n_states:(i+1)*n_states, 0:n_inputs] = A_i_B

        # Copy the previous row's blocks (shifted right)
        if i > 0:
            B_lifted[i*n_states:(i+1)*n_states, n_inputs:(i+1)*n_inputs] = \
                B_lifted[(i-1)*n_states:i*n_states, 0:i*n_inputs]

        # Update A_power for next iteration
        A_power = A_power @ Ad  # Ad^(i+1) = Ad^i @ Ad

    return A_lifted, B_lifted


def get_force_constraints(friction_coeff, mass, time_horizon, swing_foot='right'):
    """
    Generates inequality constraints for ground reaction forces based on friction limits.
    There are two forces per foot (Fx, Fy), and two feet.
    Fy has a lower bound of 0 (no pulling from the ground) and an upper bound of 1.5*mass * g
    Fx is bounded by friction limits: -mu * Fy <= Fx <= mu * Fy

    Parameters:
    - friction_coeff: float, coefficient of friction
    - mass: float, mass of the robot
    - time_horizon: int, number of time steps in the horizon

    Returns:
    - c: np.array, constraint bounds
    - C: np.array, constraint matrix
    """
    g = GRAVITY  # gravitational acceleration
    n_forces = 4  # Fx_r, Fy_r, Fx_l, Fy_l per time step
    N = time_horizon
    n_constraints_per_foot = 4  # 2 for Fy bounds, 2 for friction cone
    n_feet = 2
    total_constraints = N * n_feet * n_constraints_per_foot

    # Pre-allocate arrays efficiently
    C = np.zeros((total_constraints, n_forces * N))
    c = np.zeros(total_constraints)

    max_vertical_force = 1.5 * mass * g

    # Build constraint matrix efficiently using numpy indexing
    constraint_idx = 0
    for t in range(N):
        idx_start = t * n_forces

        # For each foot (right at idx 0,1 and left at idx 2,3)
        for foot_offset in [0, 2]:
            fx_idx = idx_start + foot_offset
            fy_idx = idx_start + foot_offset + 1

            # Check if this is the swing foot
            is_swing_foot = (swing_foot == 'right' and foot_offset == 0) or (swing_foot == 'left' and foot_offset == 2)

            if is_swing_foot:
                # For swing foot: set both Fx and Fy to be very small (near zero)
                # Fx >= 0  =>  -Fx <= 0
                C[constraint_idx, fx_idx] = -1
                c[constraint_idx] = 0
                constraint_idx += 1

                # Fx <= 0  =>  Fx <= 0
                C[constraint_idx, fx_idx] = 1
                c[constraint_idx] = 0
                constraint_idx += 1

                # Fy >= 0  =>  -Fy <= 0
                C[constraint_idx, fy_idx] = -1
                c[constraint_idx] = 0
                constraint_idx += 1

                # Fy <= 0  =>  Fy <= 0
                C[constraint_idx, fy_idx] = 1
                c[constraint_idx] = 0
                constraint_idx += 1
            else:
                # For stance foot: normal force constraints
                # Fy >= 0  =>  -Fy <= 0
                C[constraint_idx, fy_idx] = -1
                c[constraint_idx] = 0
                constraint_idx += 1

                # Fy <= 1.5 * mass * g
                C[constraint_idx, fy_idx] = 1
                c[constraint_idx] = max_vertical_force
                constraint_idx += 1

                # Fx >= -mu * Fy  =>  -Fx - mu * Fy <= 0
                C[constraint_idx, fx_idx] = -1
                C[constraint_idx, fy_idx] = -friction_coeff
                c[constraint_idx] = 0
                constraint_idx += 1

                # Fx <= mu * Fy  =>  Fx - mu * Fy <= 0
                C[constraint_idx, fx_idx] = 1
                C[constraint_idx, fy_idx] = -friction_coeff
                c[constraint_idx] = 0
                constraint_idx += 1

    return c, C

def solve_qp(x_init, x_ref, A_lifted, B_lifted, force_contraints, Q_state=None, R_input=None):
    """
    Solves the quadratic programming problem for MPC.

    Parameters:
    - x_init: np.array, initial state
    - x_ref: np.array, reference trajectory (flattened for all time steps)
    - A_lifted: np.array, lifted state transition matrix
    - B_lifted: np.array, lifted control input matrix
    - force_contraints: tuple, (c, C) inequality constraints
    - Q_state: np.array, state deviation weight matrix (n_states x n_states), defaults to identity
    - R_input: np.array, control effort weight matrix (n_inputs x n_inputs), defaults to identity

    Returns:
    - u_optimal: np.array, optimal control inputs
    """
    n_states_total = A_lifted.shape[0]  # n_states * N
    n_inputs_total = B_lifted.shape[1]  # n_inputs * N

    # Determine dimensions
    n_states = A_lifted.shape[1]  # Single time step state dimension
    n_inputs = n_inputs_total // (n_states_total // n_states)  # Single time step input dimension
    N = n_states_total // n_states  # Time horizon steps

    # Default weights if not provided
    if Q_state is None:
        Q_state_diag = np.zeros(n_states)
        Q_state_diag[0] = 2.0  # x position
        Q_state_diag[1] = 20.0  # y position
        Q_state_diag[2] = 10.0  # theta
        Q_state_diag[3] = 5.0  # x velocity
        Q_state_diag[4] = 5.0  # y velocity
        Q_state_diag[5] = 5.0  # angular velocity
        Q_state_diag[6] = 0.0  # gravity (not controlled)
        Q_state = np.diag(Q_state_diag)
    if R_input is None:
        R_input = np.eye(n_inputs)*1e-4

    # Build block-diagonal L matrix for state deviation weights over horizon
    # Apply decaying factor of 0.99 for each time step
    decay_factor = 0.99
    decay_weights = np.array([decay_factor**i for i in range(N)])
    L = np.block([[decay_weights[i] * Q_state if i == j else np.zeros((n_states, n_states))
                   for j in range(N)] for i in range(N)])

    # Build block-diagonal K matrix for control effort weights over horizon
    # Apply decaying factor of 0.99 for each time step
    K = np.block([[decay_weights[i] * R_input if i == j else np.zeros((n_inputs, n_inputs))
                   for j in range(N)] for i in range(N)])

    # Cost function: minimize (x - x_ref)^T L (x - x_ref) + u^T K u
    # where x = A_lifted @ x_init + B_lifted @ u
    H = 2*(B_lifted.T @ L @ B_lifted + K)
    g = 2*B_lifted.T @ L @ (A_lifted @ x_init - x_ref.flatten())

    # Inequality constraints
    c, C = force_contraints

    # Add regularization to H to ensure positive definiteness and improve conditioning
    H = H + 1e-5 * np.eye(H.shape[0])

    # Try OSQP first with relaxed tolerances for better numerical stability
    try:
        res = qp(H, g, C, c, solver="osqp", verbose=False,
                eps_abs=1e-4, eps_rel=1e-4, max_iter=10000, polish=True)
        if res is not None:
            return res
    except Exception as e:
        print(f"OSQP solver failed with error: {e}")
        pass  # Silently try next solver

    # Try other solvers as fallback
    solvers_to_try = ["clarabel"]
    for solver in solvers_to_try:
        try:
            res = qp(H, g, C, c, solver=solver, verbose=False)
            if res is not None:
                return res
        except Exception as e:
            print(f"{solver} solver failed with error: {e}")
            continue

    # All solvers failed - return zero control as safe fallback
    print("WARNING: QP solver failed!")
    return np.zeros(n_inputs_total)
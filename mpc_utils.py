import numpy as np
from qpsolvers import solve_qp

def generate_reference_trajectory(initial_state, desired_velocity, desired_ang_vel, time_horizon, dt):
    """
    Generates a reference trajectory for the biped robot.
    
    Parameters:
    - initial_state: np.array, initial state of the robot [x, y, theta, x_dot, y_dot, theta_dot]
    - desired_velocity: float, desired forward velocity
    - desired_ang_vel: float, desired angular velocity
    - time_horizon: float, total time for the trajectory
    - dt: float, time step
    
    Returns:
    - trajectory: np.array of shape (N, 6), where N is the number of time steps
    """
    N = time_horizon
    trajectory = np.zeros((N, len(initial_state)))
    
    for i in range(N):
        t = i * dt
        trajectory[i, 0] = initial_state[0] + desired_velocity[0] * t  # x position
        trajectory[i, 1] = initial_state[1] + desired_velocity[1] * t  # y position
        trajectory[i, 2] = initial_state[2] + desired_ang_vel * t  # theta
        trajectory[i, 3] = desired_velocity[0]                         # x velocity
        trajectory[i, 4] = desired_velocity[1]                     # y velocity
        trajectory[i, 5] = desired_ang_vel                          # angular velocity
        trajectory[i, 6] = 0                          # angular velocity
        
    return trajectory


def get_lifted_dynamics_matrices(A, B, dt):
    """
    Computes the lifted dynamics matrices for MPC efficiently by reusing matrix multiplications.
    Note: x_simulated_traj = A_lifted * x0 + B_lifted * u_sequence

    Parameters:
    - A: np.array, state transition matrix
    - B: np.array, control input matrix
    - dt: float, time step

    Returns:
    - A_lifted: np.array, lifted state transition matrix
    - B_lifted: np.array, lifted control input matrix
    """
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    N = int(1/dt)  # Assuming a fixed horizon of 1 second for simplicity

    A_lifted = np.zeros((n_states * N, n_states))
    B_lifted = np.zeros((n_states * N, n_inputs * N))

    # Incrementally build A_lifted by multiplying by A at each step
    A_power = A.copy()  # A^1
    for i in range(N):
        A_lifted[i*n_states:(i+1)*n_states, :] = A_power
        if i < N - 1:
            A_power = A_power @ A  # A^(i+2) = A^(i+1) @ A

    # Build B_lifted efficiently by reusing previous rows
    # Each row is constructed by: [A^i @ B, previous_row[:-1]]
    A_power = np.eye(n_states)  # Start with A^0 = I
    for i in range(N):
        # Compute A^i @ B for the leftmost block
        A_i_B = A_power @ B

        # Set the leftmost block
        B_lifted[i*n_states:(i+1)*n_states, 0:n_inputs] = A_i_B

        # Copy the previous row's blocks (shifted right)
        if i > 0:
            B_lifted[i*n_states:(i+1)*n_states, n_inputs:(i+1)*n_inputs] = \
                B_lifted[(i-1)*n_states:i*n_states, 0:i*n_inputs]

        # Update A_power for next iteration
        A_power = A_power @ A  # A^(i+1) = A^i @ A

    return A_lifted, B_lifted


def get_force_constraints(friction_coeff, mass, time_horizon):
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
    g = 9.81  # gravitational acceleration
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

def solve_qp(x_init, x_ref, A_lifted, B_lifted, force_contraints):
    """
    Solves the quadratic programming problem for MPC.

    Parameters:
    - x_init: np.array, initial state
    - x_ref: np.array, reference trajectory
    - A_lifted: np.array, lifted state transition matrix
    - B_lifted: np.array, lifted control input matrix

    Returns:
    - u_optimal: np.array, optimal control inputs
    """
    L = np.eye(A_lifted.shape[1])  # State deviation weight 
    K = np.eye(B_lifted.shape[1])  # Control effort weight
    
    # Cost function
    H = 2*(B_lifted.T@L@B_lifted + K)
    g = 2*B_lifted.T@L@(A_lifted@x_init - x_ref)
    
    # Inequality constraints
    c, C = force_contraints
    
    res = solve_qp(H, g, C, c, solver="clarabel", verbose=False)

    return res
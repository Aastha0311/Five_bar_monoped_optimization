import numpy as np
import biped_2D as bp
import mpc_utils as mpc

# Initialize exactly as in the simulation
robot = bp.Robot_biped_2D(mass=5.0, moment_of_inertia=0.02, L_thigh=0.2, L_shank=0.2, friction_coeff=0.8)
init_state = np.array([0, 0.3, 0, 0, 0, 0, -9.81])
foot_pos = np.array([0.1, -0.3, -0.1, -0.3])
desired_velocity = np.array([0.5, 0.0])
desired_ang_vel = 0.0
swing_foot = 'right'

print("="*60)
print("DEBUGGING FIRST QP SOLVE")
print("="*60)
print(f"\nInitial state: y={init_state[1]}m, ydot={init_state[4]}m/s")
print(f"Foot positions: R=({foot_pos[0]}, {foot_pos[1]}), L=({foot_pos[2]}, {foot_pos[3]})")
print(f"Swing foot: {swing_foot}")
print()

# Build matrices
A_mat = robot.A_hat()
B_mat = robot.B_hat(com_pos=init_state[0:3], foot_pos_world_frame=foot_pos, swing_foot=swing_foot)

print("B matrix (showing which foot can apply forces):")
print("Rows: [x, y, theta, xdot, ydot, thetadot, gravity]")
print("Cols: [Fx_r, Fy_r, Fx_l, Fy_l]")
print(B_mat)
print()

# Generate reference
ref_traj = mpc.generate_reference_trajectory(init_state, desired_velocity, desired_ang_vel, time_horizon=10, dt=0.1)
print("Reference (first 3 steps):")
for i in range(3):
    print(f"  t={i*0.1:.1f}s: x={ref_traj[i,0]:.3f}, y={ref_traj[i,1]:.3f}, theta={ref_traj[i,2]:.3f}, xdot={ref_traj[i,3]:.3f}, ydot={ref_traj[i,4]:.3f}")
print()

# Build lifted matrices
A_lifted, B_lifted = mpc.get_lifted_dynamics_matrices(A_mat, B_mat, time_horizon=10, dt=0.1)
print(f"Lifted matrices: A_lifted shape={A_lifted.shape}, B_lifted shape={B_lifted.shape}")
print()

# Get constraints
force_constraints = mpc.get_force_constraints(robot.friction_coeff, robot.mass, time_horizon=10, swing_foot=swing_foot)
c, C = force_constraints
print(f"Constraints: C shape={C.shape}, c shape={c.shape}")
print(f"First timestep constraints (first 8 rows):")
for i in range(8):
    row = C[i, :4]
    bound = c[i]
    nonzero = [(j, row[j]) for j in range(4) if abs(row[j]) > 1e-10]
    print(f"  Row {i}: {nonzero} <= {bound:.2f}")
print()

# Check if the problem is feasible at all
# For right foot swinging, we need left foot to provide ~49N vertically
print("Feasibility check:")
print(f"  Left foot needs to provide ~{robot.mass * 9.81:.2f}N vertically")
print(f"  Upper bound on left foot Fy: {1.5 * robot.mass * 9.81:.2f}N")
print(f"  Friction cone: |Fx_left| <= {robot.friction_coeff} * Fy_left")
print()

# Try solving
print("Attempting to solve QP...")
optimal_forces = mpc.solve_qp(init_state, ref_traj, A_lifted, B_lifted, force_constraints)

print(f"\nOptimal forces (first timestep):")
print(f"  Right foot (swing): Fx={optimal_forces[0]:.2f}N, Fy={optimal_forces[1]:.2f}N")
print(f"  Left foot (stance): Fx={optimal_forces[2]:.2f}N, Fy={optimal_forces[3]:.2f}N")
print(f"  Total vertical force: {optimal_forces[1] + optimal_forces[3]:.2f}N (need ~49.05N)")
print()

# Check constraint satisfaction
u_first = optimal_forces[:4]
violations = C[:8, :4] @ u_first - c[:8]
print("Constraint satisfaction (first timestep, should all be <= 0):")
for i in range(8):
    status = "✓" if violations[i] <= 1e-6 else "✗ VIOLATED"
    print(f"  Constraint {i}: {violations[i]:8.4f} {status}")

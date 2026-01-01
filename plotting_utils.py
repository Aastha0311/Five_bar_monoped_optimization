import numpy as np
import matplotlib.pyplot as plt


def inverse_kinematics_2D(com_pos, foot_pos, L_thigh, L_shank):
    """
    Calculate joint angles (hip and knee) for a 2D leg.

    Args:
        com_pos: [x, y, theta] - COM position and orientation
        foot_pos: [x, y] - foot position in world frame
        L_thigh: length of thigh
        L_shank: length of shank

    Returns:
        hip_angle: angle of thigh relative to vertical (radians)
        knee_angle: angle of knee joint (radians)
    """
    # Transform foot position to body frame
    dx = foot_pos[0] - com_pos[0]
    dy = foot_pos[1] - com_pos[1]

    # Rotate by body angle to get position in body frame
    theta = com_pos[2]
    dx_body = dx * np.cos(-theta) - dy * np.sin(-theta)
    dy_body = dx * np.sin(-theta) + dy * np.cos(-theta)

    # Distance to foot
    r = np.sqrt(dx_body**2 + dy_body**2)

    # Check if position is reachable
    if r > (L_thigh + L_shank):
        r = L_thigh + L_shank - 0.001  # Slightly less than max reach
    if r < abs(L_thigh - L_shank):
        r = abs(L_thigh - L_shank) + 0.001

    # Use law of cosines to find knee angle
    cos_knee = (L_thigh**2 + L_shank**2 - r**2) / (2 * L_thigh * L_shank)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    knee_angle = np.pi - np.arccos(cos_knee)  # Interior angle

    # Find hip angle
    alpha = np.arctan2(dx_body, -dy_body)  # Angle to foot from COM
    beta = np.arccos((L_thigh**2 + r**2 - L_shank**2) / (2 * L_thigh * r))
    hip_angle = alpha - beta

    return hip_angle, knee_angle


def calculate_joint_torques(foot_force, com_pos, foot_pos, L_thigh, L_shank):
    """
    Calculate joint torques using Jacobian transpose method: tau = J^T @ F

    Args:
        foot_force: [Fx, Fy] - force at foot in WORLD frame
        com_pos: [x, y, theta] - COM position
        foot_pos: [x, y] - foot position in world frame
        L_thigh: thigh length
        L_shank: shank length

    Returns:
        hip_torque: torque at hip joint
        knee_torque: torque at knee joint
    """
    # Get joint angles
    hip_angle, knee_angle = inverse_kinematics_2D(com_pos, foot_pos, L_thigh, L_shank)

    # Transform forces from world frame to body frame
    theta = com_pos[2]
    Fx_world = foot_force[0]
    Fy_world = foot_force[1]
    Fx_body = Fx_world * np.cos(-theta) - Fy_world * np.sin(-theta)
    Fy_body = Fx_world * np.sin(-theta) + Fy_world * np.cos(-theta)

    # Calculate Jacobian: J maps joint velocities to foot velocities
    # foot_vel = J @ [hip_dot, knee_dot]
    # For 2D leg:
    # foot_x = L_thigh * sin(hip) + L_shank * sin(hip + knee)
    # foot_y = -L_thigh * cos(hip) - L_shank * cos(hip + knee)

    # Jacobian derivatives:
    J = np.zeros((2, 2))
    J[0, 0] = L_thigh * np.cos(hip_angle) + L_shank * np.cos(hip_angle + knee_angle)  # dx/d(hip)
    J[0, 1] = L_shank * np.cos(hip_angle + knee_angle)  # dx/d(knee)
    J[1, 0] = L_thigh * np.sin(hip_angle) + L_shank * np.sin(hip_angle + knee_angle)  # dy/d(hip)
    J[1, 1] = L_shank * np.sin(hip_angle + knee_angle)  # dy/d(knee)

    # Calculate torques: tau = J^T @ F
    F_body = np.array([Fx_body, Fy_body])
    torques = J.T @ F_body

    hip_torque = torques[0]
    knee_torque = torques[1]

    return hip_torque, knee_torque


def plot_simulation_results(time_data, state_data, foot_pos_data, foot_force_data,
                           phase_data, L_thigh, L_shank):
    """
    Create comprehensive plots of the simulation results.

    Args:
        time_data: list of time values
        state_data: list of states [x, y, theta, x_dot, y_dot, theta_dot, z]
        foot_pos_data: list of foot positions [r_x, r_y, l_x, l_y]
        foot_force_data: list of foot forces [Fr_x, Fr_y, Fl_x, Fl_y]
        phase_data: list of phase values (0 to 1)
        L_thigh: thigh length
        L_shank: shank length
    """
    # Convert lists to numpy arrays
    time_data = np.array(time_data)
    state_data = np.array(state_data)
    foot_pos_data = np.array(foot_pos_data)
    foot_force_data = np.array(foot_force_data)
    phase_data = np.array(phase_data)

    # Calculate joint angles and torques
    right_hip_angles = []
    right_knee_angles = []
    left_hip_angles = []
    left_knee_angles = []
    right_hip_torques = []
    right_knee_torques = []
    left_hip_torques = []
    left_knee_torques = []

    for i in range(len(time_data)):
        com_pos = state_data[i, 0:3]
        foot_pos_r = foot_pos_data[i, 0:2]
        foot_pos_l = foot_pos_data[i, 2:4]
        foot_force_r = foot_force_data[i, 0:2]
        foot_force_l = foot_force_data[i, 2:4]

        # Right leg angles and torques
        hip_r, knee_r = inverse_kinematics_2D(com_pos, foot_pos_r, L_thigh, L_shank)
        right_hip_angles.append(hip_r)
        right_knee_angles.append(knee_r)

        torque_hip_r, torque_knee_r = calculate_joint_torques(foot_force_r, com_pos, foot_pos_r, L_thigh, L_shank)
        right_hip_torques.append(torque_hip_r)
        right_knee_torques.append(torque_knee_r)

        # Left leg angles and torques
        hip_l, knee_l = inverse_kinematics_2D(com_pos, foot_pos_l, L_thigh, L_shank)
        left_hip_angles.append(hip_l)
        left_knee_angles.append(knee_l)

        torque_hip_l, torque_knee_l = calculate_joint_torques(foot_force_l, com_pos, foot_pos_l, L_thigh, L_shank)
        left_hip_torques.append(torque_hip_l)
        left_knee_torques.append(torque_knee_l)

    # Convert to arrays and degrees
    right_hip_angles = np.rad2deg(np.array(right_hip_angles))
    right_knee_angles = np.rad2deg(np.array(right_knee_angles))
    left_hip_angles = np.rad2deg(np.array(left_hip_angles))
    left_knee_angles = np.rad2deg(np.array(left_knee_angles))
    right_hip_torques = np.array(right_hip_torques)
    right_knee_torques = np.array(right_knee_torques)
    left_hip_torques = np.array(left_hip_torques)
    left_knee_torques = np.array(left_knee_torques)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. COM Position
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(time_data, state_data[:, 0], 'b-', label='x', linewidth=1.5)
    ax1.plot(time_data, state_data[:, 1], 'r-', label='y', linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('COM Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. COM Orientation
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(time_data, np.rad2deg(state_data[:, 2]), 'g-', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (deg)')
    ax2.set_title('COM Orientation (theta)')
    ax2.grid(True, alpha=0.3)

    # 3. COM Velocity
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(time_data, state_data[:, 3], 'b-', label='x_dot', linewidth=1.5)
    ax3.plot(time_data, state_data[:, 4], 'r-', label='y_dot', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('COM Velocity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Foot Positions
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(time_data, foot_pos_data[:, 0], 'b-', label='Right X', linewidth=1.5)
    ax4.plot(time_data, foot_pos_data[:, 2], 'r-', label='Left X', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('X Position (m)')
    ax4.set_title('Foot X Positions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Foot Y Positions
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(time_data, foot_pos_data[:, 1], 'b-', label='Right Y', linewidth=1.5)
    ax5.plot(time_data, foot_pos_data[:, 3], 'r-', label='Left Y', linewidth=1.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Y Position (m)')
    ax5.set_title('Foot Y Positions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Phase
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(time_data, phase_data, 'k-', linewidth=1.5)
    ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Phase')
    ax6.set_title('Gait Phase (0-0.5: Right Swing, 0.5-1: Left Swing)')
    ax6.set_ylim([0, 1])
    ax6.grid(True, alpha=0.3)

    # 7. Joint Angles
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(time_data, right_hip_angles, 'b-', label='Right Hip', linewidth=1.5)
    ax7.plot(time_data, left_hip_angles, 'r-', label='Left Hip', linewidth=1.5)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Angle (deg)')
    ax7.set_title('Hip Angles')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Knee Angles
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(time_data, right_knee_angles, 'b-', label='Right Knee', linewidth=1.5)
    ax8.plot(time_data, left_knee_angles, 'r-', label='Left Knee', linewidth=1.5)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Angle (deg)')
    ax8.set_title('Knee Angles')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Foot Forces (magnitude)
    ax9 = plt.subplot(3, 3, 9)
    right_force_mag = np.sqrt(foot_force_data[:, 0]**2 + foot_force_data[:, 1]**2)
    left_force_mag = np.sqrt(foot_force_data[:, 2]**2 + foot_force_data[:, 3]**2)
    ax9.plot(time_data, right_force_mag, 'b-', label='Right Foot', linewidth=1.5)
    ax9.plot(time_data, left_force_mag, 'r-', label='Left Foot', linewidth=1.5)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Force (N)')
    ax9.set_title('Foot Force Magnitude')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simulation_results_main.png', dpi=150, bbox_inches='tight')
    print("Saved main plots to simulation_results_main.png")

    # Create a second figure for torques
    fig2 = plt.figure(figsize=(14, 8))

    # 1. Hip Torques
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time_data, right_hip_torques, 'b-', label='Right Hip', linewidth=1.5)
    ax1.plot(time_data, left_hip_torques, 'r-', label='Left Hip', linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Torque (Nm)')
    ax1.set_title('Hip Torques')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Knee Torques
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(time_data, right_knee_torques, 'b-', label='Right Knee', linewidth=1.5)
    ax2.plot(time_data, left_knee_torques, 'r-', label='Left Knee', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Torque (Nm)')
    ax2.set_title('Knee Torques')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Foot Forces X and Y components
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(time_data, foot_force_data[:, 0], 'b-', label='Right Fx', linewidth=1.5)
    ax3.plot(time_data, foot_force_data[:, 2], 'r-', label='Left Fx', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Force (N)')
    ax3.set_title('Foot Forces (X component)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(time_data, foot_force_data[:, 1], 'b-', label='Right Fy', linewidth=1.5)
    ax4.plot(time_data, foot_force_data[:, 3], 'r-', label='Left Fy', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Force (N)')
    ax4.set_title('Foot Forces (Y component)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simulation_results_torques.png', dpi=150, bbox_inches='tight')
    print("Saved torque plots to simulation_results_torques.png")

    plt.show()

    # Print some statistics
    print("\n=== Simulation Statistics ===")
    print(f"Simulation duration: {time_data[-1]:.2f} seconds")
    print(f"Number of data points: {len(time_data)}")
    print(f"\nCOM Position:")
    print(f"  Final X: {state_data[-1, 0]:.3f} m")
    print(f"  Final Y: {state_data[-1, 1]:.3f} m")
    print(f"  Average velocity: {np.mean(state_data[:, 3]):.3f} m/s")
    print(f"\nJoint Angle Ranges:")
    print(f"  Right Hip: {np.min(right_hip_angles):.1f}° to {np.max(right_hip_angles):.1f}°")
    print(f"  Right Knee: {np.min(right_knee_angles):.1f}° to {np.max(right_knee_angles):.1f}°")
    print(f"  Left Hip: {np.min(left_hip_angles):.1f}° to {np.max(left_hip_angles):.1f}°")
    print(f"  Left Knee: {np.min(left_knee_angles):.1f}° to {np.max(left_knee_angles):.1f}°")
    print(f"\nForce Ranges:")
    print(f"  Right Foot: Fx=[{np.min(foot_force_data[:, 0]):.2f}, {np.max(foot_force_data[:, 0]):.2f}] N, Fy=[{np.min(foot_force_data[:, 1]):.2f}, {np.max(foot_force_data[:, 1]):.2f}] N")
    print(f"  Left Foot:  Fx=[{np.min(foot_force_data[:, 2]):.2f}, {np.max(foot_force_data[:, 2]):.2f}] N, Fy=[{np.min(foot_force_data[:, 3]):.2f}, {np.max(foot_force_data[:, 3]):.2f}] N")
    print(f"\nTorque Ranges:")
    print(f"  Right Hip: {np.min(right_hip_torques):.2f} to {np.max(right_hip_torques):.2f} Nm")
    print(f"  Right Knee: {np.min(right_knee_torques):.2f} to {np.max(right_knee_torques):.2f} Nm")
    print(f"  Left Hip: {np.min(left_hip_torques):.2f} to {np.max(left_hip_torques):.2f} Nm")
    print(f"  Left Knee: {np.min(left_knee_torques):.2f} to {np.max(left_knee_torques):.2f} Nm")

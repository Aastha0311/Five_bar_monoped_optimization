import numpy as np
import matplotlib.pyplot as plt


def plot_simulation_results(time_data, state_data, foot_pos_data, foot_force_data,
                           phase_data, joint_angles_data, joint_torques_data):
    """
    Create comprehensive plots of the simulation results.

    Args:
        time_data: list of time values
        state_data: list of states [x, y, theta, x_dot, y_dot, theta_dot, z]
        foot_pos_data: list of foot positions [r_x, r_y, l_x, l_y]
        foot_force_data: list of foot forces [Fr_x, Fr_y, Fl_x, Fl_y]
        phase_data: list of phase values (0 to 1)
        joint_angles_data: list of joint angles [right_hip, right_knee, left_hip, left_knee] in radians
        joint_torques_data: list of joint torques [right_hip, right_knee, left_hip, left_knee] in Nm
    """
    # Convert lists to numpy arrays
    time_data = np.array(time_data)
    state_data = np.array(state_data)
    foot_pos_data = np.array(foot_pos_data)
    foot_force_data = np.array(foot_force_data)
    phase_data = np.array(phase_data)
    joint_angles_data = np.array(joint_angles_data)
    joint_torques_data = np.array(joint_torques_data)

    # Extract joint angles and convert to degrees
    right_hip_angles = np.rad2deg(joint_angles_data[:, 0])
    right_knee_angles = np.rad2deg(joint_angles_data[:, 1])
    left_hip_angles = np.rad2deg(joint_angles_data[:, 2])
    left_knee_angles = np.rad2deg(joint_angles_data[:, 3])

    # Extract joint torques
    right_hip_torques = joint_torques_data[:, 0]
    right_knee_torques = joint_torques_data[:, 1]
    left_hip_torques = joint_torques_data[:, 2]
    left_knee_torques = joint_torques_data[:, 3]

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

    # 7. Joint Angles - Hip
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(time_data, right_hip_angles, 'b-', label='Right Hip', linewidth=1.5)
    ax7.plot(time_data, left_hip_angles, 'r-', label='Left Hip', linewidth=1.5)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Angle (deg)')
    ax7.set_title('Hip Angles')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Joint Angles - Knee
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
    plt.savefig('logs/simulation_results_main.png', dpi=150, bbox_inches='tight')
    print("Saved main plots to logs/simulation_results_main.png")

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
    plt.savefig('logs/simulation_results_torques.png', dpi=150, bbox_inches='tight')
    print("Saved torque plots to logs/simulation_results_torques.png")

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

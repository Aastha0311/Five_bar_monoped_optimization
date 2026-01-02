import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import utils.plotting_utils as plotting_utils
from components.robot import GRAVITY, Robot_biped_2D
from components.controller import Controller


class Simulator:
    """
    Simulator for a 2D biped robot with real-time visualization.

    The simulator handles:
    - Physics integration (dynamics)
    - Gait phase management
    - Swing foot trajectory generation
    - Real-time visualization of the robot
    - Data recording for post-simulation analysis
    """

    def __init__(self, robot: Robot_biped_2D, controller: Controller, sim_dt, visualize=True):
        """
        Initialize the simulator.

        Args:
            robot: Robot model containing physical parameters
            controller: Controller that generates foot forces and swing targets
            sim_dt: Simulation timestep (seconds)
            visualize: Whether to show real-time visualization window
        """
        self.robot = robot
        self.controller = controller
        self.sim_dt = sim_dt
        self.visualize = visualize

        # Gait phase (0 to 1): 0-0.5 = right foot swings, 0.5-1 = left foot swings
        self.phase = 0

        # Initialize state: [x, y, theta, x_dot, y_dot, theta_dot, y_ddot]
        self.state = self._initialize_state()

        # Initialize foot positions in world frame: [right_x, right_y, left_x, left_y]
        self.foot_pos_world_frame = self._initialize_feet()

        # Storage for swing foot trajectories
        self._swing_start_pos = {
            'right': np.array([self.foot_pos_world_frame[0], self.foot_pos_world_frame[1]]),
            'left': np.array([self.foot_pos_world_frame[2], self.foot_pos_world_frame[3]])
        }

        # Data recording
        self.current_time = 0.0
        self.time_history = []
        self.state_history = []
        self.foot_pos_history = []
        self.foot_force_history = []
        self.phase_history = []
        self.joint_angles_history = []  # [right_hip, right_knee, left_hip, left_knee]
        self.joint_torques_history = []  # [right_hip, right_knee, left_hip, left_knee]

        # Current joint angles (cached to avoid recalculation in render)
        self.current_joint_angles = None  # [right_hip, right_knee, left_hip, left_knee]

        # Visualization setup
        if self.visualize:
            self._setup_visualization()

    def _initialize_state(self):
        """Initialize robot state with COM at nominal height."""
        return np.array([
            0,                               # x position
            self.robot.nominal_com_height,   # y position
            0,                               # theta (orientation)
            0,                               # x velocity
            0,                               # y velocity
            0,                               # theta velocity
            -GRAVITY                         # y acceleration
        ])

    def _initialize_feet(self):
        """Initialize feet on ground with narrow stance."""
        stance_width = 0.05
        return np.array([
            -stance_width, self.robot.ground_level,  # right foot
            -stance_width, self.robot.ground_level   # left foot
        ])

    def _setup_visualization(self):
        """Initialize the real-time visualization window."""
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Biped Robot Simulation')

        # Initialize plot elements (will be updated in render)
        self.torso_patch = None
        self.right_thigh_line = None
        self.right_shank_line = None
        self.left_thigh_line = None
        self.left_shank_line = None
        self.right_knee_circle = None
        self.left_knee_circle = None
        self.ground_line = None
        self.phase_text = None
        self.time_text = None

    def simulate_the_robot(self, foot_force, swing_foot_pos, swing_foot):
        """
        Perform one simulation step: update physics and swing foot trajectory.

        Args:
            foot_force: [F_right_x, F_right_y, F_left_x, F_left_y] - forces from controller
            swing_foot_pos: [right_x, right_y, left_x, left_y] - target positions
            swing_foot: 'right' or 'left' - which foot is currently swinging
        """
        # Record current state before updating
        self._record_data(foot_force)

        # Update physics (integrate dynamics)
        self._update_dynamics(foot_force, swing_foot)

        # Update swing foot trajectory
        self._update_swing_foot(swing_foot_pos)

        # Update time
        self.current_time += self.sim_dt

        # Render visualization if enabled
        if self.visualize:
            self.render()

    def _record_data(self, foot_force):
        """Record current simulation data for later analysis."""
        self.time_history.append(self.current_time)
        self.state_history.append(self.state.copy())
        self.foot_pos_history.append(self.foot_pos_world_frame.copy())
        self.foot_force_history.append(foot_force.copy())
        self.phase_history.append(self.phase)

        # Calculate and record joint angles (store for use in render)
        com_pos = self.state[0:3]
        right_hip, right_knee = self.robot.inverse_kinematics(
            com_pos, self.foot_pos_world_frame[0:2]
        )
        left_hip, left_knee = self.robot.inverse_kinematics(
            com_pos, self.foot_pos_world_frame[2:4]
        )

        # Cache current angles to avoid recalculation in render
        self.current_joint_angles = [right_hip, right_knee, left_hip, left_knee]
        self.joint_angles_history.append(self.current_joint_angles.copy())

        # Calculate and record joint torques
        right_hip_torque, right_knee_torque = self.robot.calculate_joint_torques(
            foot_force[0:2], com_pos, self.foot_pos_world_frame[0:2]
        )
        left_hip_torque, left_knee_torque = self.robot.calculate_joint_torques(
            foot_force[2:4], com_pos, self.foot_pos_world_frame[2:4]
        )
        self.joint_torques_history.append([right_hip_torque, right_knee_torque,
                                          left_hip_torque, left_knee_torque])

    def _update_dynamics(self, foot_force, swing_foot):
        """Update robot state using dynamics integration."""
        com_pos = self.state[0:3]
        self.state = self.robot.step_lti(
            self.state,
            foot_force,
            com_pos,
            self.foot_pos_world_frame,
            swing_foot,
            self.sim_dt
        )

    def _update_swing_foot(self, swing_foot_pos):
        """Update swing foot position and manage phase transitions."""
        old_phase = self.phase
        phase_increment = self.sim_dt * self.controller.stepping_frequency
        self.phase = (self.phase + phase_increment) % 1.0

        # Handle phase transitions
        self._handle_phase_transitions(old_phase, swing_foot_pos)

        # Interpolate swing foot trajectory
        self._interpolate_swing_trajectory(swing_foot_pos)

    def _handle_phase_transitions(self, old_phase, swing_foot_pos):
        """
        Detect and handle phase transitions (when swing foot changes).
        Snap finished swing foot to target and capture new swing start position.
        """
        # Right foot swing starts (phase crosses 0) - left foot finished
        if old_phase > 0.5 and self.phase < 0.5:
            self.foot_pos_world_frame[2] = swing_foot_pos[2]  # Snap left foot x
            self.foot_pos_world_frame[3] = swing_foot_pos[3]  # Snap left foot y
            self._swing_start_pos['right'] = np.array([
                self.foot_pos_world_frame[0],
                self.foot_pos_world_frame[1]
            ])

        # Left foot swing starts (phase crosses 0.5) - right foot finished
        elif old_phase < 0.5 and self.phase >= 0.5:
            self.foot_pos_world_frame[0] = swing_foot_pos[0]  # Snap right foot x
            self.foot_pos_world_frame[1] = swing_foot_pos[1]  # Snap right foot y
            self._swing_start_pos['left'] = np.array([
                self.foot_pos_world_frame[2],
                self.foot_pos_world_frame[3]
            ])

    def _interpolate_swing_trajectory(self, swing_foot_pos):
        """
        Interpolate swing foot position with parabolic trajectory for height.
        """
        if self.phase < 0.5:
            # Right foot swings
            swing_phase = self.phase / 0.5
            initial_pos = self._swing_start_pos['right']
            target_pos = swing_foot_pos[0:2]

            # Parabolic swing height (peaks at swing_phase = 0.5)
            swing_height_mult = 4 * swing_phase * (1 - swing_phase)

            self.foot_pos_world_frame[0] = initial_pos[0] + (target_pos[0] - initial_pos[0]) * swing_phase
            self.foot_pos_world_frame[1] = (initial_pos[1] + (target_pos[1] - initial_pos[1]) * swing_phase +
                                           self.controller.swing_height * swing_height_mult)
        else:
            # Left foot swings
            swing_phase = (self.phase - 0.5) / 0.5
            initial_pos = self._swing_start_pos['left']
            target_pos = swing_foot_pos[2:4]

            # Parabolic swing height
            swing_height_mult = 4 * swing_phase * (1 - swing_phase)

            self.foot_pos_world_frame[2] = initial_pos[0] + (target_pos[0] - initial_pos[0]) * swing_phase
            self.foot_pos_world_frame[3] = (initial_pos[1] + (target_pos[1] - initial_pos[1]) * swing_phase +
                                           self.controller.swing_height * swing_height_mult)

    def render(self):
        """Render the current robot state in the visualization window."""
        if not self.visualize:
            return

        self.ax.clear()

        # Get current state
        com_x, com_y, theta = self.state[0:3]

        # Use cached joint angles (calculated in _record_data)
        if self.current_joint_angles is None:
            return  # No data yet

        right_hip, right_knee, left_hip, left_knee = self.current_joint_angles

        # Calculate knee positions for visualization
        right_knee_x = com_x + self.robot.L_thigh * np.sin(right_hip)
        right_knee_y = com_y - self.robot.L_thigh * np.cos(right_hip)
        left_knee_x = com_x + self.robot.L_thigh * np.sin(left_hip)
        left_knee_y = com_y - self.robot.L_thigh * np.cos(left_hip)

        # Draw ground
        x_min = com_x - 1.5
        x_max = com_x + 1.5
        self.ax.plot([x_min, x_max], [0, 0], 'k-', linewidth=2, label='Ground')

        # Draw torso (as a rotated rectangle)
        torso_width = 0.15
        torso_height = 0.1

        # Create rectangle corners and rotate them manually for better control
        # Corners in body frame (centered at origin)
        corners = np.array([
            [-torso_width/2, -torso_height/2],
            [torso_width/2, -torso_height/2],
            [torso_width/2, torso_height/2],
            [-torso_width/2, torso_height/2],
            [-torso_width/2, -torso_height/2]  # Close the rectangle
        ])

        # Rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                   [sin_theta, cos_theta]])

        # Rotate and translate corners to world frame
        rotated_corners = (rotation_matrix @ corners.T).T
        world_corners = rotated_corners + np.array([com_x, com_y])

        # Draw the rotated torso
        self.ax.fill(world_corners[:, 0], world_corners[:, 1],
                    color='gray', alpha=0.7, edgecolor='black', linewidth=2, label='Torso')

        # Draw an orientation indicator (arrow) on the torso to visualize rotation
        # The arrow points "up" relative to the torso orientation
        arrow_length = 0.12
        arrow_end_x = com_x + arrow_length * np.sin(theta)  # Points "up" in body frame
        arrow_end_y = com_y + arrow_length * np.cos(theta)
        self.ax.arrow(com_x, com_y, arrow_end_x - com_x, arrow_end_y - com_y,
                     head_width=0.04, head_length=0.025, fc='orange', ec='darkorange',
                     linewidth=3, zorder=10, label='Orientation')

        # Draw right leg (RED)
        self._draw_leg(
            com_x, com_y,
            right_knee_x, right_knee_y,
            self.foot_pos_world_frame[0],
            self.foot_pos_world_frame[1],
            color='red',
            label='Right Leg'
        )

        # Draw left leg (BLUE)
        self._draw_leg(
            com_x, com_y,
            left_knee_x, left_knee_y,
            self.foot_pos_world_frame[2],
            self.foot_pos_world_frame[3],
            color='blue',
            label='Left Leg'
        )

        # Set axis limits to follow robot
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(-0.1, 0.8)

        # Add info text with more precision for theta to show it's changing
        phase_str = "Right Swing" if self.phase < 0.5 else "Left Swing"
        info_text = (f"Time: {self.current_time:.2f}s | Phase: {phase_str} ({self.phase:.2f})\n"
                    f"Theta: {np.degrees(theta):.4f}° | COM: ({com_x:.3f}, {com_y:.3f})\n"
                    f"Velocity: ({self.state[3]:.3f}, {self.state[4]:.3f}) m/s")
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Biped Robot Simulation')
        self.ax.legend(loc='upper right')

        plt.pause(0.001)  # Small pause to update display

    def _draw_leg(self, hip_x, hip_y, knee_x, knee_y, foot_x, foot_y, color, label):
        """
        Draw a single leg with thigh, shank, and knee joint.

        Args:
            hip_x, hip_y: Hip position
            knee_x, knee_y: Knee position
            foot_x, foot_y: Foot position
            color: Color for the leg
            label: Label for legend
        """

        # Draw thigh (hip to knee)
        self.ax.plot([hip_x, knee_x], [hip_y, knee_y],
                    color=color, linewidth=4, label=f'{label} Thigh')

        # Draw shank (knee to foot)
        self.ax.plot([knee_x, foot_x], [knee_y, foot_y],
                    color=color, linewidth=4, linestyle='--', label=f'{label} Shank')

        # Draw knee joint as circle
        knee_circle = plt.Circle((knee_x, knee_y), 0.02, color=color,
                                fill=True, zorder=5)
        self.ax.add_patch(knee_circle)

        # Draw foot as small circle
        foot_circle = plt.Circle((foot_x, foot_y), 0.015, color=color,
                                fill=True, zorder=5)
        self.ax.add_patch(foot_circle)

    def plot_graphs(self):
        """Generate comprehensive plots of the simulation results."""
        if len(self.time_history) == 0:
            print("No data to plot. Run the simulation first.")
            return

        plotting_utils.plot_simulation_results(
            time_data=self.time_history,
            state_data=self.state_history,
            foot_pos_data=self.foot_pos_history,
            foot_force_data=self.foot_force_history,
            phase_data=self.phase_history,
            joint_angles_data=self.joint_angles_history,
            joint_torques_data=self.joint_torques_history
        )

    def close(self):
        """Close the visualization window."""
        if self.visualize:
            plt.close(self.fig)

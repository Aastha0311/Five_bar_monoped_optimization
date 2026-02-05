import numpy as np
import matplotlib.pyplot as plt
from components.robot_monoped import GRAVITY, Robot_monoped_2D
from components.controller_monoped import Controller
from utils import plotting_utils_monoped

class Simulator:
    def __init__(self,
                 robot: Robot_monoped_2D,
                 controller: Controller,
                 sim_dt,
                 visualize=True,
                 stance_fraction=0.6):

        self.robot = robot
        self.controller = controller
        self.sim_dt = sim_dt
        self.visualize = visualize
        self.stance_fraction = stance_fraction

        self.phase = 0.0

        # State: [x, y, theta, xdot, ydot, thetadot, g]
        self.state = np.array([
            0.0,
            robot.nominal_com_height,
            0.0,
            0.0,
            0.0,
            0.0,
            -GRAVITY
        ])

        self.foot_pos_world_frame = np.array([
            0.0,
            self.robot.ground_level
        ])

        self._swing_start_pos = self.foot_pos_world_frame.copy()
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

        if visualize:
            self._setup_visualization()

    # --------------------------------------------------
    # Phase logic
    # --------------------------------------------------

    def _setup_visualization(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Monoped Robot Simulation')

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

    # def simulate_the_robot(self, foot_force, swing_foot_pos, swing_foot):
    #     """
    #     Perform one simulation step: update physics and swing foot trajectory.

    #     Args:
    #         foot_force: [F_right_x, F_right_y, F_left_x, F_left_y] - forces from controller
    #         swing_foot_pos: [right_x, right_y, left_x, left_y] - target positions
    #         swing_foot: 'right' or 'left' - which foot is currently swinging
    #     """
    #     # Record current state before updating
    #     self._record_data(foot_force)

    #     # Update physics (integrate dynamics)
    #     self._update_dynamics(foot_force, swing_foot)

    #     # Update swing foot trajectory
    #     self._update_swing_foot(swing_foot_pos)

    #     # Update time
    #     self.current_time += self.sim_dt

    #     # Render visualization if enabled
    #     if self.visualize:
    #         self.render()
    def simulate_the_robot(self, foot_force, swing_foot_pos, in_stance):
        """
        One simulation step with correct hybrid logic.
        """

        # 1. Record
        self._record_data(foot_force)

        # 2. Enforce stance foot constraint BEFORE dynamics
        if in_stance:
            self.foot_pos_world_frame[1] = self.robot.ground_level

        # 3. Integrate COM dynamics
        self._update_dynamics(foot_force, in_stance)

        # 4. Ground safety for COM (ABSOLUTELY REQUIRED)
        min_com_height = self.robot.ground_level + 0.05
        if self.state[1] < min_com_height:
            self.state[1] = min_com_height
            if self.state[4] < 0.0:
                self.state[4] = 0.0

        # 5. Update swing foot ONLY in flight
        if not in_stance:
            self._update_swing_foot(swing_foot_pos)

        # 6. Time update
        self.current_time += self.sim_dt

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
        hip_angle, knee_angle = self.robot.inverse_kinematics(
            com_pos, self.foot_pos_world_frame
        )
        

        # Cache current angles to avoid recalculation in render
        self.current_joint_angles = [hip_angle, knee_angle]
        self.joint_angles_history.append(self.current_joint_angles.copy())

        # Calculate and record joint torques
        hip_torque, knee_torque = self.robot.calculate_joint_torques(
            foot_force, com_pos, self.foot_pos_world_frame
        )
        
        self.joint_torques_history.append([hip_torque, knee_torque])

    def _update_dynamics(self, foot_force, in_stance):
        """Update robot state using dynamics integration."""
        com_pos = self.state[0:3]
        self.state = self.robot.step_lti(
            self.state,
            foot_force,
            com_pos,
            self.foot_pos_world_frame,
            in_stance,
            self.sim_dt
        )
    # def _update_swing_foot(self, swing_foot_pos):
    #     """Update swing foot position and manage phase transitions."""
    #     old_phase = self.phase
    #     phase_increment = self.sim_dt * self.controller.stepping_frequency
    #     self.phase = (self.phase + phase_increment) % 1.0

    #     # Handle phase transitions
    #     self._handle_phase_transitions(old_phase, swing_foot_pos)

    #     # Interpolate swing foot trajectory
    #     self._interpolate_swing_trajectory(swing_foot_pos)

    def _update_swing_foot(self, swing_foot_pos):
        """
        Flight-phase foot interpolation only.
        """
        swing_phase = (self.phase - self.stance_fraction) / (1.0 - self.stance_fraction)
        swing_phase = np.clip(swing_phase, 0.0, 1.0)

        initial_pos = self._swing_start_pos
        target_pos = swing_foot_pos

        h_mult = 4 * swing_phase * (1 - swing_phase)

        self.foot_pos_world_frame[0] = (
            initial_pos[0]
            + (target_pos[0] - initial_pos[0]) * swing_phase
        )
        self.foot_pos_world_frame[1] = (
            self.robot.ground_level
            + self.controller.swing_height * h_mult
        )




    # def in_stance(self):
    #     old_phase = self.phase
    #     phase_increment = self.sim_dt * self.controller.stepping_frequency
    #     self.phase = (self.phase + phase_increment) % 1.0
    #     return self.phase < self.stance_fraction
    
    def handle_phase_transition(self, old_phase):
        if old_phase < self.stance_fraction and self.phase >= self.stance_fraction:
            # stance → flight
            self._swing_start_pos = self.foot_pos_world_frame.copy()

        if old_phase >= self.stance_fraction and self.phase < self.stance_fraction:
            # flight → stance
            self.foot_pos_world_frame[1] = self.robot.ground_level
            


    def _interpolate_swing_trajectory(self, swing_foot_pos):
        """
        Interpolate swing foot position with parabolic trajectory for height.
        """
        if self.phase > self.stance_fraction:
            # Right foot swings
            swing_phase = (self.phase - self.stance_fraction) / (1.0 - self.stance_fraction)
            initial_pos = self._swing_start_pos
            target_pos = swing_foot_pos

            # Parabolic swing height (peaks at swing_phase = 0.5)
            swing_height_mult = 4 * swing_phase * (1 - swing_phase)

            self.foot_pos_world_frame[0] = initial_pos[0] + (target_pos[0] - initial_pos[0]) * swing_phase
            self.foot_pos_world_frame[1] = (initial_pos[1] + (target_pos[1] - initial_pos[1]) * swing_phase +
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

        hip_angle, knee_angle = self.current_joint_angles

        # Calculate knee positions for visualization
        knee_x = com_x + self.robot.L_thigh * np.sin(hip_angle)
        knee_y = com_y - self.robot.L_thigh * np.cos(hip_angle)

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
            knee_x, knee_y,
            self.foot_pos_world_frame[0],
            self.foot_pos_world_frame[1],
            color='red',
            label='Leg'
        )

        

        # Set axis limits to follow robot
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(-1, 1)

        # Add info text with more precision for theta to show it's changing
        phase_str = "Stance" if self.phase < self.stance_fraction else "Swing"
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
        self.ax.set_title('Monoped Robot Simulation')
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

        plotting_utils_monoped.plot_simulation_results(
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



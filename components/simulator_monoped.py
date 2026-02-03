import numpy as np
import matplotlib.pyplot as plt
from components.robot_monoped import GRAVITY, Robot_monoped_2D
from components.controller_monoped import Controller


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
            robot.ground_level
        ])

        self._swing_start_pos = self.foot_pos_world_frame.copy()
        self.time = 0.0

        self.time_history = []
        self.state_history = []
        self.foot_pos_history = []
        self.phase_history = []

        if visualize:
            self._setup_visualization()

    # --------------------------------------------------
    # Phase logic
    # --------------------------------------------------
    def in_stance(self):
        return self.phase < self.stance_fraction

    # --------------------------------------------------
    # Main step
    # --------------------------------------------------
    def step(self, desired_velocity, desired_ang_vel):

        in_stance = self.in_stance()

        foot_force, target_foot_pos = self.controller.walk(
            init_state=self.state,
            foot_pos_world_frame=self.foot_pos_world_frame,
            desired_velocity=desired_velocity,
            desired_ang_vel=desired_ang_vel,
            in_stance=in_stance
        )

        # Integrate centroidal dynamics
        self.state = self.robot.step_lti(
            self.state,
            foot_force,
            self.state[0:3],
            self.foot_pos_world_frame,
            in_stance,
            self.sim_dt
        )

        # -------------------------------
        # CRITICAL FIX: stance projection
        # -------------------------------
        if in_stance:
            max_leg = self.robot.L_thigh + self.robot.L_shank
            dx = self.state[0] - self.foot_pos_world_frame[0]

            # Project COM back to valid height
            desired_y = np.sqrt(
                max(max_leg**2 - dx**2, 0.05**2)
            ) + self.robot.ground_level

            self.state[1] = max(self.state[1], desired_y)

            # Kill downward velocity in stance
            if self.state[4] < 0.0:
                self.state[4] = 0.0

        # Update foot
        self._update_foot(target_foot_pos)

        # Phase update
        self.phase = (self.phase + self.sim_dt * self.controller.stepping_frequency) % 1.0
        self.time += self.sim_dt

        self._log()

        if self.visualize:
            self.render()

    # --------------------------------------------------
    # Foot motion
    # --------------------------------------------------
    def _update_foot(self, target_pos):
        if self.in_stance():
            self.foot_pos_world_frame[1] = self.robot.ground_level
            self._swing_start_pos = self.foot_pos_world_frame.copy()
        else:
            flight_phase = (self.phase - self.stance_fraction) / (1.0 - self.stance_fraction)
            flight_phase = np.clip(flight_phase, 0.0, 1.0)

            h = 4 * flight_phase * (1 - flight_phase)

            self.foot_pos_world_frame[0] = (
                self._swing_start_pos[0]
                + (target_pos[0] - self._swing_start_pos[0]) * flight_phase
            )
            self.foot_pos_world_frame[1] = (
                self.robot.ground_level
                + self.controller.swing_height * h
            )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    def _log(self):
        self.time_history.append(self.time)
        self.state_history.append(self.state.copy())
        self.foot_pos_history.append(self.foot_pos_world_frame.copy())
        self.phase_history.append(self.phase)

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    def _setup_visualization(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def render(self):
        self.ax.clear()

        com_x, com_y = self.state[0], self.state[1]
        foot_x, foot_y = self.foot_pos_world_frame

        hip, knee = self.robot.inverse_kinematics(
            self.state[0:3], self.foot_pos_world_frame
        )

        knee_x = com_x + self.robot.L_thigh * np.sin(hip)
        knee_y = com_y - self.robot.L_thigh * np.cos(hip)

        self.ax.plot([com_x - 1, com_x + 1], [0, 0], 'k-', lw=2)
        self.ax.plot([com_x, knee_x], [com_y, knee_y], 'b-', lw=4)
        self.ax.plot([knee_x, foot_x], [knee_y, foot_y], 'b--', lw=4)

        self.ax.plot(foot_x, foot_y, 'ro')
        self.ax.plot(knee_x, knee_y, 'ko')

        self.ax.set_xlim(com_x - 1, com_x + 1)
        self.ax.set_ylim(-0.1, 0.8)
        self.ax.set_aspect('equal')

        self.ax.set_title(
            f"Time: {self.time:.2f}s | Phase: {self.phase:.2f} | "
            f"{'STANCE' if self.in_stance() else 'FLIGHT'}"
        )

        self.ax.grid(True)
        plt.pause(0.001)



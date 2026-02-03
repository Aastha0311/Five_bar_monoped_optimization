import numpy as np
import utils.mpc_utils_monoped as mpc
from components.robot_monoped import GRAVITY, Robot_monoped_2D


class Controller:
    def __init__(self, robot: Robot_monoped_2D,
                 time_horizon,
                 stepping_frequency,
                 control_dt,
                 raibert_gain):

        self.robot = robot
        self.time_horizon = time_horizon
        self.control_dt = control_dt
        self.stepping_frequency = stepping_frequency
        self.Kp = raibert_gain

        self.swing_height = 0.1  # flight apex clearance

    # -------------------------------------------------
    # Standing controller (single foot)
    # -------------------------------------------------
    def stand(self, state, foot_pos_world_frame, desired_height):
        g = GRAVITY
        weight = self.robot.mass * g

        # Simple PD on height via vertical GRF
        Fy = weight + 0.3 * (desired_height - state[1])
        Fx = 0.0

        foot_force = np.array([Fx, Fy])
        return foot_force, foot_pos_world_frame

    # -------------------------------------------------
    # Walking / hopping controller
    # -------------------------------------------------
    def walk(self,
             init_state,
             foot_pos_world_frame,
             desired_velocity,
             desired_ang_vel,
             in_stance):
        """
        in_stance = 1 → stance phase (MPC active)
        in_stance = 0 → flight phase (Raibert active)
        """

        # ===============================
        # STANCE PHASE: Convex MPC
        # ===============================
        if in_stance:
            A_mat = self.robot.A_hat()
            B_mat = self.robot.B_hat(
                com_pos=init_state[0:3],
                foot_pos_world_frame=foot_pos_world_frame,
                in_stance=1
            )

            reference_trajectory = mpc.generate_reference_trajectory(
                init_state,
                desired_velocity,
                desired_ang_vel,
                self.time_horizon,
                self.control_dt,
                desired_height=self.robot.nominal_com_height
            )

            A_lifted, B_lifted = mpc.get_lifted_dynamics_matrices(
                A_mat, B_mat, self.time_horizon, dt=self.control_dt
            )

            force_constraints = mpc.get_force_constraints(
                self.robot.friction_coeff,
                self.robot.mass,
                self.time_horizon
            )

            optimal_force = mpc.solve_qp(
                init_state,
                reference_trajectory,
                A_lifted,
                B_lifted,
                force_constraints
            )

            # Only first control input is applied
            foot_force = optimal_force[:2]
            optimal_foot_pos = foot_pos_world_frame

        # ===============================
        # FLIGHT PHASE: Raibert foot placement
        # ===============================
        else:
            foot_force = np.array([0.0, 0.0])

            T = 1.0 / self.stepping_frequency

            expected_foot_x = (
                init_state[0]
                + 0.5 * init_state[3] * T
            )

            raibert_correction = -self.Kp * (desired_velocity[0] - init_state[3])
            expected_foot_x += raibert_correction

            optimal_foot_pos = np.array([
                expected_foot_x,
                self.robot.ground_level
            ])

        return foot_force, optimal_foot_pos

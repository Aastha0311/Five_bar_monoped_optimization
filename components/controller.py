import numpy as np
import utils.mpc_utils as mpc
from components.robot import GRAVITY, Robot_biped_2D

class Controller:
    def __init__(self, robot: Robot_biped_2D, time_horizon, stepping_frequency, control_dt, raibert_gain):
        self.robot = robot
        self.time_horizon = time_horizon
        self.control_dt = control_dt
        self.stepping_frequency = stepping_frequency 
        self.Kp = raibert_gain
        self.swing_height = 0.1

    def stand(self, state, foot_pos_world_frame, desired_height):
        g = GRAVITY
        weight_of_robot = self.robot.mass * g
        foot_force_per_foot = weight_of_robot / 2 + 0.3 * (desired_height - state[1])
        foot_force = np.array([0, foot_force_per_foot, 0, foot_force_per_foot])
        return foot_force, foot_pos_world_frame
    
    def walk(self, init_state, foot_pos_world_frame, desired_velocity, desired_ang_vel, swing_foot):

        ## Stance foot
        A_mat = self.robot.A_hat()
        B_mat = self.robot.B_hat(com_pos=init_state[0:3], foot_pos_world_frame=foot_pos_world_frame, swing_foot=swing_foot)

        reference_trajectory = mpc.generate_reference_trajectory(init_state, desired_velocity, desired_ang_vel, self.time_horizon, self.control_dt, desired_height=self.robot.nominal_com_height)
        A_lifted, B_lifted = mpc.get_lifted_dynamics_matrices(A_mat, B_mat, self.time_horizon, dt=self.control_dt)
        force_constraints = mpc.get_force_constraints(self.robot.friction_coeff, self.robot.mass, self.time_horizon, swing_foot=swing_foot)
        optimal_foot_force = mpc.solve_qp(init_state, reference_trajectory, A_lifted, B_lifted, force_constraints)

        # Debug output - print first 10 steps to see failure point
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1

        ## Swing foot
        expected_foot_x = init_state[0] + 0.5*init_state[3] * 1/self.stepping_frequency
        raiberts_correction = -(desired_velocity[0] - init_state[3])*self.Kp
        expected_foot_x += raiberts_correction

        if swing_foot == 'right':
            optimal_foot_pos = np.array([expected_foot_x, self.robot.ground_level, foot_pos_world_frame[2], foot_pos_world_frame[3]])
        else:
            optimal_foot_pos = np.array([foot_pos_world_frame[0], foot_pos_world_frame[1], expected_foot_x, self.robot.ground_level])

        return optimal_foot_force[:4], optimal_foot_pos




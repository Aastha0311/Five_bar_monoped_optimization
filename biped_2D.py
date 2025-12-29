import numpy as np
import mpc_utils as mpc
class Robot_biped_2D:
    def __init__(self, mass=5.0, moment_of_inertia=0.02, L_thigh=0.2, L_shank=0.2, friction_coeff=0.8):
        self.mass    = mass
        self.moment_of_inertia = moment_of_inertia 
        self.L_thigh = L_thigh
        self.L_shank = L_shank
        self.friction_coeff = friction_coeff

        # com_pos: [x(t), y(t), theta(t)]
        # com_vel: [x_dot(t), y_dot(t), theta_dot(t)]
        # action:  [F_r_x, F_r_y, F_l_x, F_l_y]
        # foot_pos_body_frame: [p_r_x, p_r_y, p_l_x, p_l_y]

        self.com_pos = np.array([0,0,0])
        self.com_vel = np.array([0,0,0])
        self.action  = np.array([0,0,0,0])

    def A_hat(self):
        A_hat = np.array([[0,0,0,1,0,0,0],
                          [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,-1],
                          [0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0]])
        return A_hat
    
    def B_hat(self, com_pos, foot_pos_world_frame, swing_foot='right'):
        m_inv = 1/self.mass
        I = self.moment_of_inertia

        p_c_x = com_pos[0]
        p_c_y = com_pos[1]

        p_r_x = foot_pos_world_frame[0]
        p_r_y = foot_pos_world_frame[1]
        p_l_x = foot_pos_world_frame[2]
        p_l_y = foot_pos_world_frame[3]

        if swing_foot == 'right':
            I_right = 0
            I_left  = 1
        else:
            I_right = 1
            I_left  = 0

        B_hat = np.array([[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [m_inv,0,m_inv,0],
                          [0,m_inv,0,m_inv],
                          [(-(p_r_y - p_c_y)/I)*I_right,(p_r_x - p_c_x)/I*I_right,(-(p_l_y - p_c_y)/I)*I_left,(p_l_x - p_c_x)/I*I_left],
                          [0,0,0,0]])
        return B_hat

    def dynamics(self, X, u, com_pos, foot_pos_world_frame):
        A = self.A_hat()
        B = self.B_hat(com_pos=com_pos, foot_pos_world_frame=foot_pos_world_frame)
        X_dot = np.dot(A,X) + np.dot(B,u)

        return X_dot

class controller:
    def __init__(self, robot: Robot_biped_2D=None, time_horizon=10, stepping_frequency=0.1, dt=0.1):
        self.robot = robot
        self.time_horizon = time_horizon
        self.dt = dt
        self.stepping_frequency = stepping_frequency 
        self.Kp = 0.15

    def stand(self, foot_pos_world_frame):
        g = 9.81
        weight_of_robot = self.robot.mass * g
        action = np.array([0, weight_of_robot/2, 0, weight_of_robot/2])
        return action, foot_pos_world_frame
    
    def walk(self, init_state, foot_pos_world_frame, desired_velocity, desired_ang_vel, swing_foot = 'right'):

        ## Stance foot
        A_mat = self.robot.A_hat()
        B_mat = self.robot.B_hat(com_pos=init_state[0:3], foot_pos_world_frame=foot_pos_world_frame, swing_foot=swing_foot)

        reference_trajectory = mpc.generate_reference_trajectory(init_state, desired_velocity, desired_ang_vel, self.time_horizon, self.dt)
        A_lifted, B_lifted = mpc.get_lifted_dynamics_matrices(A_mat, B_mat, dt=self.dt)
        force_constraints = mpc.get_force_constraints(self.robot.friction_coeff, self.robot.mass, self.time_horizon)
        optimal_action = mpc.solve_qp(init_state, reference_trajectory, A_lifted, B_lifted, force_constraints)

        ## Swing foot
        expected_foot_x = init_state[0] + 0.5*init_state[3] * 1/self.stepping_frequency
        raiberts_correction = -(desired_velocity[0] - init_state[3])*self.Kp
        expected_foot_x += raiberts_correction
        if swing_foot == 'right':
            optimal_foot_pos = np.array([expected_foot_x, -0.3, foot_pos_world_frame[2], foot_pos_world_frame[3]])
        else:
            optimal_foot_pos = np.array([foot_pos_world_frame[0], foot_pos_world_frame[1], expected_foot_x, -0.3])


        return optimal_action[:4], optimal_foot_pos



class simulate:
    def __init__(self):
        self.robot      = Robot_biped_2D()
        self.controller = controller()

    def simulate_the_robot(self,t):
        pass



if __name__ == "__main__":
    robot = Robot_biped_2D(mass=5.0, moment_of_inertia=0.02, L_thigh=0.2, L_shank=0.2, friction_coeff=0.8)
    controller = controller(robot)

    init_state = np.array([0,0,0,0,0,0,0])
    foot_pos_world_frame = np.array([0.1, -0.3, -0.1, -0.3])  # right foot (x,y), left foot (x,y)
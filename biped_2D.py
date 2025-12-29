import numpy as np

class Robot_biped_2D:
    def __init__(self):
        self.mass    = 5.0 # kg
        self.moment_of_inertia = 0.02 
        self.L_thigh = 0.2 # m
        self.L_shank = 0.2 # m

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
    
    def B_hat(self, com_pos, foot_pos_world_frame):
        m_inv = 1/self.mass
        I = self.moment_of_inertia

        p_c_x = com_pos[0]
        p_c_y = com_pos[1]

        p_r_x = foot_pos_world_frame[0]
        p_r_y = foot_pos_world_frame[1]
        p_l_x = foot_pos_world_frame[2]
        p_l_y = foot_pos_world_frame[3]

        B_hat = np.array([[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [m_inv,0,m_inv,0],
                          [0,m_inv,0,m_inv],
                          [(-(p_r_y - p_c_y)/I),(p_r_x - p_c_x)/I,(-(p_l_y - p_c_y)/I),(p_l_x - p_c_x)/I],
                          [0,0,0,0]])
        return B_hat

    def dynamics(self, X, u, com_pos, foot_pos_world_frame):
        A = self.A_hat()
        B = self.B_hat(com_pos=com_pos, foot_pos_world_frame=foot_pos_world_frame)
        X_dot = np.dot(A,X) + np.dot(B,u)

        return X_dot

class controller:
    def __init__(self):
        self.STANCE_PHASE = 1
        self.SWING_PHASE  = 0

        self.STAND_GAIT = 0

    def gait_generator(self, t, gait):
        STANCE_PHASE = self.STANCE_PHASE
        SWING_PHASE  = self.SWING_PHASE
        if(gait==self.STAND_GAIT):
            return np.array([STANCE_PHASE, STANCE_PHASE])
    
    def stand_controller(self, mass_of_robot):
        g = 9.81
        weight_of_robot = mass_of_robot * g
        action = np.array([0, weight_of_robot/2, 0, weight_of_robot/2])
        return action

class simulate:
    def __init__(self):
        self.robot      = Robot_biped_2D()
        self.controller = controller()

    def simulate_the_robot(self,t):
        
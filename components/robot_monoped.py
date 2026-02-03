import numpy as np
from scipy.linalg import expm

GRAVITY = 9.81


class Robot_monoped_2D:
    def __init__(self, mass, moment_of_inertia, L_thigh, L_shank, friction_coeff, density):
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.L_thigh = L_thigh
        self.L_shank = L_shank
        self.friction_coeff = friction_coeff
        self.density = density

        # Leg geometry
        self.leg_length = L_thigh + L_shank
        self.nominal_com_height = 0.85 * self.leg_length
        self.ground_level = 0.0

        # --- Mass distribution (monoped assumption) ---
        leg_mass_fraction = 0.25  # single leg ~25% total mass
        leg_mass = self.mass * leg_mass_fraction

        self.m_thigh = leg_mass * (L_thigh / self.leg_length)
        self.m_shank = leg_mass * (L_shank / self.leg_length)

        # State
        # X = [x, y, theta, xdot, ydot, thetadot, g]
        self.com_pos = np.zeros(3)
        self.com_vel = np.zeros(3)

        # Action: [Fx, Fy] at the single foot
        self.action = np.zeros(2)

    # -------------------------------------------------
    # Continuous-time centroidal dynamics
    # -------------------------------------------------
    def A_hat(self):
        return np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])

    def B_hat(self, com_pos, foot_pos_world_frame, in_stance):
        """
        in_stance = 1 → stance phase (contact active)
        in_stance = 0 → flight phase (no contact)
        """
        m_inv = 1 / self.mass
        I = self.moment_of_inertia

        p_c_x, p_c_y = com_pos[0], com_pos[1]
        p_f_x, p_f_y = foot_pos_world_frame[0], foot_pos_world_frame[1]

        return np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [m_inv * in_stance, 0],
            [0, m_inv * in_stance],
            [-(p_f_y - p_c_y) / I * in_stance,
              (p_f_x - p_c_x) / I * in_stance],
            [0, 0]
        ])

    # -------------------------------------------------
    # Discrete-time LTI step (exact discretization)
    # -------------------------------------------------
    def step_lti(self, X, u, com_pos, foot_pos_world_frame, in_stance, dt):
        A = self.A_hat()
        B = self.B_hat(com_pos, foot_pos_world_frame, in_stance)

        n, m = A.shape[0], B.shape[1]
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A
        M[:n, n:] = B

        Md = expm(M * dt)
        Ad = Md[:n, :n]
        Bd = Md[:n, n:]

        return Ad @ X + Bd @ u

    # -------------------------------------------------
    # Inverse kinematics (single leg)
    # -------------------------------------------------
    def inverse_kinematics(self, com_pos, foot_pos):
        dx = foot_pos[0] - com_pos[0]
        dy = foot_pos[1] - com_pos[1]

        distance = np.sqrt(dx**2 + dy**2)
        max_reach = self.L_thigh + self.L_shank
        min_reach = abs(self.L_thigh - self.L_shank)

        distance = np.clip(distance, min_reach * 1.01, max_reach * 0.99)

        cos_knee = (self.L_thigh**2 + self.L_shank**2 - distance**2) / \
                   (2 * self.L_thigh * self.L_shank)
        cos_knee = np.clip(cos_knee, -1.0, 1.0)
        knee_angle = np.pi - np.arccos(cos_knee)

        alpha = np.arctan2(dx, -dy)
        beta = np.arccos(
            np.clip(
                (self.L_thigh**2 + distance**2 - self.L_shank**2) /
                (2 * self.L_thigh * distance),
                -1.0, 1.0
            )
        )
        hip_angle = alpha - beta

        return hip_angle, knee_angle

    # -------------------------------------------------
    # Gravity torques
    # -------------------------------------------------
    def calculate_gravitational_torques(self, hip_angle, knee_angle):
        g = GRAVITY

        shank_angle = hip_angle + (np.pi - knee_angle)

        tau_hip_thigh = self.m_thigh * g * (self.L_thigh / 2) * np.sin(hip_angle)

        shank_com_x = (
            self.L_thigh * np.sin(hip_angle)
            + (self.L_shank / 2) * np.sin(shank_angle)
        )
        tau_hip_shank = self.m_shank * g * shank_com_x

        hip_torque = tau_hip_thigh + tau_hip_shank
        knee_torque = self.m_shank * g * (self.L_shank / 2) * np.sin(shank_angle)

        return hip_torque, knee_torque

    # -------------------------------------------------
    # Joint torques from GRF + gravity
    # -------------------------------------------------
    def calculate_joint_torques(self, foot_force, com_pos, foot_pos):
        hip_angle, knee_angle = self.inverse_kinematics(com_pos, foot_pos)
        shank_angle = hip_angle + (np.pi - knee_angle)

        theta = com_pos[2]
        Fx, Fy = foot_force

        Fx_body = Fx * np.cos(-theta) - Fy * np.sin(-theta)
        Fy_body = Fx * np.sin(-theta) + Fy * np.cos(-theta)

        J = np.zeros((2, 2))
        J[0, 0] = self.L_thigh * np.cos(hip_angle) + self.L_shank * np.cos(shank_angle)
        J[0, 1] = -self.L_shank * np.cos(shank_angle)
        J[1, 0] = self.L_thigh * np.sin(hip_angle) + self.L_shank * np.sin(shank_angle)
        J[1, 1] = -self.L_shank * np.sin(shank_angle)

        torques_grf = J.T @ np.array([Fx_body, Fy_body])

        tau_g_hip, tau_g_knee = self.calculate_gravitational_torques(
            hip_angle, knee_angle
        )

        return (
            torques_grf[0] + tau_g_hip,
            torques_grf[1] + tau_g_knee
        )

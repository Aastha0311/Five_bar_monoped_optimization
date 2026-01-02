import numpy as np
from scipy.linalg import expm

GRAVITY = 9.81
class Robot_biped_2D:
    def __init__(self, mass, moment_of_inertia, L_thigh, L_shank, friction_coeff, density):
        self.mass    = mass
        self.moment_of_inertia = moment_of_inertia
        self.L_thigh = L_thigh
        self.L_shank = L_shank
        self.friction_coeff = friction_coeff
        self.density = density  # kg/m^3, assumed constant for simplicity

        # Compute nominal COM height based on leg geometry
        # Use 85% of leg length to leave margin for leg extension
        self.leg_length = L_thigh + L_shank
        self.nominal_com_height = self.leg_length * 0.85
        self.ground_level = 0.0  # Ground at y=0

        # Calculate leg segment masses based on density and link lengths
        # Assume legs are cylindrical with cross-sectional area A
        # Total leg mass is proportional to total leg length
        # We need to derive the cross-sectional area from total mass and density

        # Assume both legs together comprise a certain fraction of total mass
        # For a biped: legs ≈ 30% of total mass (15% per leg)
        leg_mass_fraction = 0.15  # One leg = 15% of total robot mass
        single_leg_mass = self.mass * leg_mass_fraction

        # Mass is proportional to length (assuming constant cross-section and density)
        # m = density × volume = density × A × L
        # For each segment: m_segment = (L_segment / L_total) × m_leg_total
        self.m_thigh = single_leg_mass * (L_thigh / self.leg_length)
        self.m_shank = single_leg_mass * (L_shank / self.leg_length)

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
                          [0,0,0,0,0,0,1],
                          [0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0]])
        return A_hat
    
    def B_hat(self, com_pos, foot_pos_world_frame, swing_foot):
        m_inv = 1/self.mass
        I = self.moment_of_inertia

        p_c_x = com_pos[0]
        p_c_y = com_pos[1]

        p_r_x = foot_pos_world_frame[0]
        p_r_y = foot_pos_world_frame[1]
        p_l_x = foot_pos_world_frame[2]
        p_l_y = foot_pos_world_frame[3]

        if swing_foot == 'right':
            I_right = 0  # Right foot in swing, doesn't apply forces
            I_left  = 1  # Left foot in stance, applies forces
        else:
            I_right = 1  # Right foot in stance, applies forces
            I_left  = 0  # Left foot in swing, doesn't apply forces

        B_hat = np.array([[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [m_inv*I_right, 0, m_inv*I_left, 0],  # Only stance foot affects x acceleration
                          [0, m_inv*I_right, 0, m_inv*I_left],  # Only stance foot affects y acceleration
                          [(-(p_r_y - p_c_y)/I)*I_right, (p_r_x - p_c_x)/I*I_right, (-(p_l_y - p_c_y)/I)*I_left, (p_l_x - p_c_x)/I*I_left],
                          [0,0,0,0]])
        return B_hat

    def dynamics(self, X, u, com_pos, foot_pos_world_frame):
        A = self.A_hat()
        B = self.B_hat(com_pos=com_pos, foot_pos_world_frame=foot_pos_world_frame)
        X_dot = np.dot(A,X) + np.dot(B,u)

        return X_dot
    
    def step_lti(self, X, u, com_pos, foot_pos_world_frame, swing_foot, dt):
        A = self.A_hat()
        B = self.B_hat(com_pos=com_pos, foot_pos_world_frame=foot_pos_world_frame, swing_foot=swing_foot)

        n = A.shape[0]
        m = B.shape[1]

        M = np.zeros((n + m, n + m))
        M[:n, :n] = A
        M[:n, n:] = B

        Md = expm(M * dt)

        Ad = Md[:n, :n]
        Bd = Md[:n, n:]

        return Ad @ X + Bd @ u

    def inverse_kinematics(self, com_pos, foot_pos):
        """
        Calculate joint angles (hip and knee) for a 2D leg using inverse kinematics.

        Args:
            com_pos: [x, y, theta] - COM position and orientation
            foot_pos: [x, y] - foot position in world frame

        Returns:
            hip_angle: angle of thigh relative to vertical (radians)
            knee_angle: angle of knee joint (radians)
        """
        # Vector from COM to foot in world frame
        dx = foot_pos[0] - com_pos[0]
        dy = foot_pos[1] - com_pos[1]

        # Distance to foot
        distance = np.sqrt(dx**2 + dy**2)

        # Check if position is reachable
        max_reach = self.L_thigh + self.L_shank
        min_reach = abs(self.L_thigh - self.L_shank)
        if distance > max_reach:
            distance = max_reach * 0.99
        if distance < min_reach:
            distance = min_reach * 1.01

        # Law of cosines for knee angle
        cos_knee = (self.L_thigh**2 + self.L_shank**2 - distance**2) / (2 * self.L_thigh * self.L_shank)
        cos_knee = np.clip(cos_knee, -1.0, 1.0)
        knee_angle = np.pi - np.arccos(cos_knee)  # Interior angle

        # Hip angle calculation
        alpha = np.arctan2(dx, -dy)  # Angle to foot from COM
        beta = np.arccos(np.clip(
            (self.L_thigh**2 + distance**2 - self.L_shank**2) / (2 * self.L_thigh * distance),
            -1.0, 1.0
        ))
        hip_angle = alpha - beta

        return hip_angle, knee_angle

    def calculate_gravitational_torques(self, hip_angle, knee_angle):
        """
        Calculate joint torques due to gravity acting on leg segments.

        Args:
            hip_angle: Hip joint angle (radians, from vertical downward)
            knee_angle: Knee joint angle (radians, interior angle)

        Returns:
            hip_torque_gravity: Gravitational torque at hip joint (Nm)
            knee_torque_gravity: Gravitational torque at knee joint (Nm)
        """
        g = GRAVITY

        # Convert knee interior angle to shank absolute angle
        # knee_angle is interior angle, so exterior angle = π - knee_angle
        # Shank absolute orientation = hip_angle + (π - knee_angle)
        shank_angle = hip_angle + (np.pi - knee_angle)

        # Hip torque from gravity
        # Torque from thigh: τ = m_thigh × g × (L_thigh/2) × sin(hip_angle)
        # The perpendicular distance from hip to thigh COM is (L_thigh/2) × sin(hip_angle)
        tau_hip_thigh = self.m_thigh * g * (self.L_thigh / 2) * np.sin(hip_angle)

        # Torque from shank about hip:
        # Shank COM position in world frame:
        # x_com = L_thigh × sin(hip_angle) + (L_shank/2) × sin(shank_angle)
        shank_com_x = self.L_thigh * np.sin(hip_angle) + (self.L_shank / 2) * np.sin(shank_angle)
        # Perpendicular distance is the horizontal component (since gravity is vertical)
        tau_hip_shank = self.m_shank * g * shank_com_x

        hip_torque_gravity = tau_hip_thigh + tau_hip_shank

        # Knee torque from gravity
        # Only the shank contributes torque about the knee
        # Perpendicular distance from knee to shank COM is (L_shank/2) × sin(shank_angle)
        knee_torque_gravity = self.m_shank * g * (self.L_shank / 2) * np.sin(shank_angle)

        return hip_torque_gravity, knee_torque_gravity

    def calculate_joint_torques(self, foot_force, com_pos, foot_pos):
        """
        Calculate total joint torques from ground reaction forces and gravity.

        Combines:
        1. Torques from ground reaction forces (via Jacobian transpose)
        2. Torques from gravity acting on leg segments

        Args:
            foot_force: [Fx, Fy] - force at foot in world frame
            com_pos: [x, y, theta] - COM position
            foot_pos: [x, y] - foot position in world frame

        Returns:
            hip_torque: total torque at hip joint (Nm)
            knee_torque: total torque at knee joint (Nm)
        """
        # Get joint angles
        hip_angle, knee_angle = self.inverse_kinematics(com_pos, foot_pos)

        # Convert knee interior angle to shank absolute angle for Jacobian
        # knee_angle is interior angle, shank absolute = hip + (π - knee)
        shank_angle = hip_angle + (np.pi - knee_angle)

        # 1. Calculate torques from ground reaction forces
        # Transform forces from world frame to body frame
        theta = com_pos[2]
        Fx_world = foot_force[0]
        Fy_world = foot_force[1]
        Fx_body = Fx_world * np.cos(-theta) - Fy_world * np.sin(-theta)
        Fy_body = Fx_world * np.sin(-theta) + Fy_world * np.cos(-theta)

        # Calculate Jacobian: J maps joint velocities to foot velocities
        # Foot position kinematics:
        # foot_x = L_thigh * sin(hip_angle) + L_shank * sin(shank_angle)
        # foot_y = -L_thigh * cos(hip_angle) - L_shank * cos(shank_angle)
        # where shank_angle = hip_angle + (π - knee_angle)
        #
        # Taking derivatives with respect to joint angles (hip_angle, knee_angle):
        # d(shank_angle)/d(hip_angle) = 1
        # d(shank_angle)/d(knee_angle) = -1
        J = np.zeros((2, 2))
        J[0, 0] = self.L_thigh * np.cos(hip_angle) + self.L_shank * np.cos(shank_angle)  # dx/d(hip)
        J[0, 1] = -self.L_shank * np.cos(shank_angle)  # dx/d(knee) [note the minus sign!]
        J[1, 0] = self.L_thigh * np.sin(hip_angle) + self.L_shank * np.sin(shank_angle)  # dy/d(hip)
        J[1, 1] = -self.L_shank * np.sin(shank_angle)  # dy/d(knee) [note the minus sign!]

        # Torques from ground reaction forces: tau_grf = J^T @ F
        F_body = np.array([Fx_body, Fy_body])
        torques_grf = J.T @ F_body

        # 2. Calculate torques from gravity
        hip_torque_gravity, knee_torque_gravity = self.calculate_gravitational_torques(hip_angle, knee_angle)

        # 3. Total torques = ground reaction + gravity
        hip_torque_total = torques_grf[0] + hip_torque_gravity
        knee_torque_total = torques_grf[1] + knee_torque_gravity

        return hip_torque_total, knee_torque_total


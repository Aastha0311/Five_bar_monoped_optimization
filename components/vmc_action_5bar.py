import mujoco as mj
import numpy as np


class Controller:

    def __init__(self, m, d, xml_path, action, thigh_length, calf_length, hip_offset, ori_l=10, ori_theta=0.0):

        self.m = m
        self.d = d

        # controller parameters from action
        self.K = action[0]
        self.C = action[1]
        self.T_gain = action[2]

        self.thigh_length = thigh_length
        self.calf_length = calf_length
        self.hip_offset_full = hip_offset
        self.ori_l = ori_l
        self.ori_theta = ori_theta
        # Joint IDs
        self.hip_left_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_left")
        self.hip_right_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_right")

        self.hip_left_dof = m.jnt_dofadr[self.hip_left_id]
        self.hip_right_dof = m.jnt_dofadr[self.hip_right_id]

        # Sites
        self.left_tip_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "left_tip")
        self.right_tip_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "right_tip")

        self.base_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "base")

    # ---------------------------------------------------
    # Positions
    # ---------------------------------------------------

    def ee_pos(self):
        left = self.d.site_xpos[self.left_tip_id]
        right = self.d.site_xpos[self.right_tip_id]
        return 0.5 * (left + right)

    def base_pos(self):
        return self.d.xipos[self.base_body_id]

    # ---------------------------------------------------
    # Leg length
    # ---------------------------------------------------

    def distance(self):
        return np.linalg.norm(self.ee_pos() - self.base_pos())

    # ---------------------------------------------------
    # Orientation
    # ---------------------------------------------------

    def equivalent_orientation(self):

        ee = self.ee_pos()
        base = self.base_pos()

        leg_vec = ee - base
        v = leg_vec / np.linalg.norm(leg_vec)

        return np.arccos(np.clip(np.dot(v, np.array([0, 0, -1])), -1, 1))

    # ---------------------------------------------------
    # Linear spring-damper force
    # ---------------------------------------------------

    def total_linear_force(self):

        l = self.distance()

        Jp = np.zeros((3, self.m.nv))
        mj.mj_jacSite(self.m, self.d, Jp, None, self.left_tip_id)

        vel = Jp @ self.d.qvel

        leg_vec = self.ee_pos() - self.base_pos()
        leg_dir = leg_vec / np.linalg.norm(leg_vec)

        ldot = leg_dir @ vel

        return self.K * (self.ori_l - l) - self.C * ldot

    # ---------------------------------------------------
    # World force
    # ---------------------------------------------------

    def force_world(self):

        base = self.base_pos()
        ee = self.ee_pos()

        leg_vec = ee - base
        l = np.linalg.norm(leg_vec)

        leg_dir = leg_vec / l

        Fl = self.total_linear_force()
        F_lin = Fl * leg_dir
        #print("Fl:", Fl, "F_lin:", F_lin)
        alpha = np.arctan2(leg_dir[0], -leg_dir[2])
        tau_t = self.T_gain * (self.ori_theta - alpha)

        leg_perp = np.array([-leg_dir[2], 0.0, leg_dir[0]])
        F_tor = (tau_t / l) * leg_perp
        #print("alpha:", alpha, "tau_t:", tau_t, "F_tor:", F_tor)
        #print("X torque:", F_tor[0], "Y torque:", F_tor[1], "Z torque:", F_tor[2])
        F = -F_lin - F_tor
        total_mass = float(np.sum(self.m.body_mass))
        F_grav = 0*total_mass * self.m.opt.gravity
        #print("total_mass:", total_mass, "gravity:", self.m.opt.gravity, "F_grav:", F_grav)
        return F + F_grav

    # ---------------------------------------------------
    # Ground contact
    # ---------------------------------------------------

    # def get_ground_contact_forces(self):

    #     ground_id = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, "floor")

    #     contact_list = []

    #     for i in range(self.d.ncon):

    #         con = self.d.contact[i]

    #         if con.geom1 == ground_id or con.geom2 == ground_id:

    #             force = np.zeros(6)
    #             mj.mj_contactForce(self.m, self.d, i, force)

    #             contact_list.append((force[:3], con))

    #     return contact_list
    def fivebar_jacobian(self,theta1, theta2, x, y, base_distance, l1, l2):
        """
        Analytical Jacobian of symmetric 5-bar mechanism

        Inputs:
            theta1, theta2 : hip angles
            x, y           : end-effector position (from MuJoCo)
            d              : base distance
            l1, l2         : link lengths

        Returns:
            J (2x2): maps [theta_dot] → [x_dot, y_dot]
        """

        # Base positions
        A = np.array([-base_distance / 2, 0.0])
        B = np.array([base_distance / 2, 0.0])

        # Elbow positions
        C1 = A + l1 * np.array([np.cos(theta1), np.sin(theta1)])
        C2 = B + l1 * np.array([np.cos(theta2), np.sin(theta2)])

        # Vectors from elbows to foot
        r1 = np.array([x, y]) - C1
        r2 = np.array([x, y]) - C2

        # ----- A matrix -----
        A_mat = np.array([
            [r1[0], r1[1]],
            [r2[0], r2[1]]
        ])

        # ----- B matrix -----
        # velocity of elbows
        dC1_dtheta1 = l1 * np.array([-np.sin(theta1), np.cos(theta1)])
        dC2_dtheta2 = l1 * np.array([-np.sin(theta2), np.cos(theta2)])

        B_mat = np.array([
            [np.dot(r1, dC1_dtheta1), 0],
            [0, np.dot(r2, dC2_dtheta2)]
        ])

        # ----- Jacobian -----
        J = np.linalg.solve(A_mat, B_mat)   # more stable than inv

        return J
    
    
    def get_ground_contact_forces(self):

        ground_id = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, "floor")

        foot_left_id = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, "foot_left")
        foot_right_id = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, "foot_right")

        contact_list = []

        for i in range(self.d.ncon):

            con = self.d.contact[i]

            # check if this is ground contact
            if con.geom1 == ground_id or con.geom2 == ground_id:

                # check if the OTHER geom is one of the feet
                other_geom = con.geom2 if con.geom1 == ground_id else con.geom1

                if other_geom in [foot_left_id, foot_right_id]:

                    force = np.zeros(6)
                    mj.mj_contactForce(self.m, self.d, i, force)

                    contact_list.append((force[:3], con))

        return contact_list

    # ---------------------------------------------------
    # Joint torque mapping
    # ---------------------------------------------------

    def joint_torque(self):

        Jp_left = np.zeros((3, self.m.nv))
        Jp_right = np.zeros((3, self.m.nv))

        mj.mj_jacSite(self.m, self.d, Jp_left, None, self.left_tip_id)
        mj.mj_jacSite(self.m, self.d, Jp_right, None, self.right_tip_id)
        
        
        Jp = 0.5 * (Jp_left + Jp_right)
        
        F = self.force_world()
        
        tau_full = Jp.T @ F
        
        tau_left = tau_full[self.hip_left_dof]
        tau_right = tau_full[self.hip_right_dof]
        

        return np.array([tau_left, tau_right])
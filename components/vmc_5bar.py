# import mujoco as mj
# import numpy as np
# import xml.etree.ElementTree as ET


# class Controller:

#     def __init__(self, m, d, xml_path, ori_l=1, ori_theta=0.0):

#         self.m = m
#         self.d = d
#         self.xml_path = xml_path

#         self.ori_l = ori_l
#         self.ori_theta = ori_theta

#         # -------------------------
#         # Joint IDs
#         # -------------------------

#         self.hip_left_id  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_left")
#         self.knee_left_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "knee_left")

#         self.hip_right_id  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_right")
#         self.knee_right_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "knee_right")

#         self.rootz_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_z")

#         self.base_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "base")
#         self.left_tip_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "left_tip")
#         self.l1, self.l2 = self._extract_link_lengths_from_xml(xml_path)

#     # ================================================================
#     # BASIC STATES
#     # ================================================================

#     def hip_left(self):
#         return self.d.qpos[self.hip_left_id]

#     def knee_left(self):
#         return self.d.qpos[self.knee_left_id]

#     def hip_right(self):
#         return self.d.qpos[self.hip_right_id]

#     def knee_right(self):
#         return self.d.qpos[self.knee_right_id]

#     def base_inertial_pos(self):
#         return self.d.xipos[self.base_body_id]

#     # ================================================================
#     # LINK LENGTHS FROM XML
#     # ================================================================

#     def thigh_length(self):
#         geom_id = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, "thigh_left")
#         fromto = self.m.geom_data[geom_id][:6]
#         p0 = fromto[:3]
#         p1 = fromto[3:6]
#         return np.linalg.norm(p1 - p0)


#     def calf_length(self):
#         geom_id = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, "shank_left")
#         fromto = self.m.geom_data[geom_id][:6]
#         p0 = fromto[:3]
#         p1 = fromto[3:6]
#         return np.linalg.norm(p1 - p0)
    
#     def _extract_link_lengths_from_xml(self, xml_path):
#         tree = ET.parse(xml_path)
#         root = tree.getroot()

#         l1 = None
#         l2 = None

#         for worldbody in root.findall("worldbody"):
#             for body in worldbody.findall("body"):
#                 if body.get("name") == "root":
#                     for base in body.findall("body"):
#                         if base.get("name") == "base":

#                             # LEFT BRANCH
#                             for link1 in base.findall("body"):
#                                 if link1.get("name") == "l1_left":

#                                     for geom in link1.findall("geom"):
#                                         if geom.get("name") == "thigh_left":
#                                             fromto = list(map(float, geom.get("fromto").split()))
#                                             p0 = fromto[:3]
#                                             p1 = fromto[3:]
#                                             l1 = np.linalg.norm(np.array(p1) - np.array(p0))

#                                     for link2 in link1.findall("body"):
#                                         if link2.get("name") == "l2_left":
#                                             for geom in link2.findall("geom"):
#                                                 if geom.get("name") == "shank_left":
#                                                     fromto = list(map(float, geom.get("fromto").split()))
#                                                     p0 = fromto[:3]
#                                                     p1 = fromto[3:]
#                                                     l2 = np.linalg.norm(np.array(p1) - np.array(p0))

#         if l1 is None or l2 is None:
#             raise ValueError("Could not extract link lengths from XML.")

#         return l1, l2

#     # ================================================================
#     # END EFFECTOR POSITION (AVERAGED)
#     # ================================================================

#     # def ee_pos(self):

#     #     l1 = self.thigh_length()
#     #     l2 = self.calf_length()

#     #     q1L = self.hip_left()
#     #     q2L = self.knee_left()

#     #     q1R = self.hip_right()
#     #     q2R = self.knee_right()

#     #     base = self.base_inertial_pos()

#     #     # Left foot
#     #     xL = base[0] + l1*np.sin(q1L) + l2*np.sin(q1L + q2L)
#     #     zL = base[2] - l1*np.cos(q1L) - l2*np.cos(q1L + q2L)

#     #     # Right foot
#     #     xR = base[0] + l1*np.sin(q1R) + l2*np.sin(q1R + q2R)
#     #     zR = base[2] - l1*np.cos(q1R) - l2*np.cos(q1R + q2R)

#     #     return np.array([(xL+xR)/2, 0, (zL+zR)/2])

#     def ee_pos(self):
#         left_site_id = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_SITE, "left_tip")
#         return self.d.site_xpos[left_site_id].copy()


#     # ================================================================
#     # DISTANCE & ORIENTATION
#     # ================================================================

#     def distance(self):
#         return np.linalg.norm(self.ee_pos() - self.base_inertial_pos())

#     def equivalent_orientation(self):
#         base = self.base_inertial_pos()
#         ee = self.ee_pos()
#         v = (ee - base) / self.distance()
#         return np.arccos(np.clip(np.dot(v, np.array([0,0,-1])), -1, 1))

#     # ================================================================
#     # SPRING-DAMPER FORCES
#     # ================================================================

#     def linear_spring_force(self, action):
#         kp = action[0]
#         return kp * (self.ori_l - self.distance())

#     # def linear_damper_force(self, action):
#     #     kd = action[1]

#     #     J = self.jacobian()
#     #     qdot = np.array([
#     #         self.d.qvel[self.hip_left_id],
#     #         self.d.qvel[self.knee_left_id],
#     #         self.d.qvel[self.hip_right_id],
#     #         self.d.qvel[self.knee_right_id]
#     #     ])

#     #     v = J @ qdot
#     #     ldot = (self.ee_pos() - self.base_inertial_pos()) @ v / self.distance()

#     #     return kd * ldot
    
#     def linear_damper_force(self,action):
#         kd = action[1]

#         J = self.jacobian()
#         qdot = np.array([
#             self.d.qvel[self.hip_left_id],
#             self.d.qvel[self.knee_left_id],
#             self.d.qvel[self.hip_right_id],
#             self.d.qvel[self.knee_right_id]
#         ])

#         v = J @ qdot
        
#         ee = self.ee_pos()
#         base = self.base_inertial_pos()

#         # Project to XZ plane
#         ee_xz = np.array([ee[0], ee[2]])
#         base_xz = np.array([base[0], base[2]])

#         leg_vec = ee_xz - base_xz
#         leg_dir = leg_vec / np.linalg.norm(leg_vec)

#         ldot = leg_dir @ v

#         return kd*ldot
    
#     def torsional_spring_force(self, action):
#         kp = action[2]
#         #return kp * (self.ori_theta - self.equivalent_orientation())
#         return 0

#     def total_linear_force(self, action):
#         return self.linear_spring_force(action) - self.linear_damper_force(action)

#     # ================================================================
#     # CARTESIAN FORCE
#     # ================================================================

#     def force_applied_ground(self, action):

#         theta = self.equivalent_orientation()
#         F = self.total_linear_force(action)
#         T = self.torsional_spring_force(action)
#         L = self.distance()

#         fx = -F*np.sin(theta) - (T/L)*np.cos(theta)
#         fz = -F*np.cos(theta) + (T/L)*np.sin(theta)
#         fx = 0
#         fz = F
        

#         return np.array([fx, fz])

#     # ================================================================
#     # 5-BAR JACOBIAN (2x4)
#     # ================================================================

#     def jacobian(self):

#         # l1 = self.thigh_length()
#         # l2 = self.calf_length()
#         l1 = self.l1
#         l2 = self.l2

#         q1L = self.hip_left()
#         q2L = self.knee_left()
#         q1R = self.hip_right()
#         q2R = self.knee_right()

#         # Left
#         J11 = l1*np.cos(q1L) + l2*np.cos(q1L + q2L)
#         J12 = l2*np.cos(q1L + q2L)
#         J21 = l1*np.sin(q1L) + l2*np.sin(q1L + q2L)
#         J22 = l2*np.sin(q1L + q2L)

#         # Right
#         J13 = l1*np.cos(q1R) + l2*np.cos(q1R + q2R)
#         J14 = l2*np.cos(q1R + q2R)
#         J23 = l1*np.sin(q1R) + l2*np.sin(q1R + q2R)
#         J24 = l2*np.sin(q1R + q2R)

#         return np.array([
#             [J11, J12, J13, J14],
#             [J21, J22, J23, J24]
#         ])
    
#     def mujoco_jacobian(self):

#         site_id = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_SITE, "left_tip")

#         Jp = np.zeros((3, self.m.nv))
#         mj.mj_jacSite(self.m, self.d, Jp, None, site_id)

#         return Jp
    
#     def get_ground_contact_forces(self):
#         """
#         Returns list of (contact_force, geom_name)
#         for all contacts involving the ground.
#         contact_force is 3D world-frame force vector.
#         """

#         # m = self.m
#         # d = self.d

#         ground_id = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, "floor")

#         contact_list = []

#         for i in range(self.d.ncon):

#             con = self.d.contact[i]

#             if con.geom1 == ground_id or con.geom2 == ground_id:

#                 force = np.zeros(6)
#                 mj.mj_contactForce(self.m, self.d, i, force)

#                 contact_force = force[:3]

#                 other_geom = con.geom2 if con.geom1 == ground_id else con.geom1
#                 geom_name = mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_GEOM, other_geom)

#                 contact_list.append((contact_force, geom_name))

#         return contact_list


#     # ================================================================
#     # TORQUE MAPPING (HIPS ONLY)
#     # ================================================================

#     def joint_torque(self, action):

#         J = self.jacobian()
#         J = self.mujoco_jacobian()

#         hip_left_dof  = self.m.jnt_dofadr[self.hip_left_id]
#         hip_right_dof = self.m.jnt_dofadr[self.hip_right_id]

#         J_reduced = np.array([
#             [J[0, hip_left_dof],  J[0, hip_right_dof]],
#             [J[2, hip_left_dof],  J[2, hip_right_dof]]
#         ])
        
        

#         F = self.force_applied_ground(action)

#         tau_full = J_reduced.T @ F

#         tau_hip_left  = tau_full[0]
#         tau_hip_right = tau_full[1]

#         return np.array([tau_hip_left, tau_hip_right])

import mujoco as mj
import numpy as np

class Controller:

    def __init__(self, m, d, xml_path, ori_l=10.0, ori_theta=0.2):
        self.m = m
        self.d = d
        self.ori_l = ori_l
        self.ori_theta = ori_theta

        # Joint IDs
        self.hip_left_id  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_left")
        self.hip_right_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_right")

        self.hip_left_dof  = m.jnt_dofadr[self.hip_left_id]
        self.hip_right_dof = m.jnt_dofadr[self.hip_right_id]

        # Sites
        self.left_tip_id  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "left_tip")
        self.right_tip_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "right_tip")

        self.base_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "base")

    # ---------------------------------------------------
    # End Effector (Midpoint for symmetry)
    # ---------------------------------------------------

    def ee_pos(self):
        left  = self.d.site_xpos[self.left_tip_id]
        right = self.d.site_xpos[self.right_tip_id]
        return 0.5 * (left + right)

    def base_pos(self):
        return self.d.xipos[self.base_body_id]

    # ---------------------------------------------------
    # Distance and Orientation (SIGNED)
    # ---------------------------------------------------

    def distance(self):
        return np.linalg.norm(self.ee_pos() - self.base_pos())

    def equivalent_orientation(self):
        ee = self.ee_pos()
        base = self.base_pos()
        leg_vec = self.ee_pos() - self.base_pos()
        v = (ee - base) / self.distance()
        return np.arccos(np.clip(np.dot(v, np.array([0,0,-1])), -1, 1))
        # alpha = np.arctan2(leg_vec[0], -leg_vec[2])
        # return alpha

    # ---------------------------------------------------
    # Spring Forces
    # ---------------------------------------------------

    def total_linear_force(self, action):
        K = action[0]
        C = action[1]

        l = self.distance()

        # compute ldot via jacobian
        Jp = np.zeros((3, self.m.nv))
        mj.mj_jacSite(self.m, self.d, Jp, None, self.left_tip_id)

        Jxz = Jp[[0,2], :]
        qdot = self.d.qvel

        vel = Jxz @ qdot
        leg_vec = self.ee_pos() - self.base_pos()
        leg_dir = leg_vec[[0,2]] / np.linalg.norm(leg_vec[[0,2]])

        ldot = leg_dir @ vel

        return K*(self.ori_l - l) - C*ldot

    # ---------------------------------------------------
    # Force in World Frame
    # ---------------------------------------------------

    # def force_world(self, action):

    #     F_l = self.total_linear_force(action)
    #     alpha = self.equivalent_orientation()
    #     l = self.distance()

    #     T = action[2] * (self.ori_theta - alpha)

    #     Fx = -F_l*np.sin(alpha) - (T/l)*np.cos(alpha)
    #     mass = mj.mj_getTotalmass(self.m)
    #     weight = mass*9.81
    #     Fz = -F_l*np.cos(alpha) + (T/l)*np.sin(alpha) - weight
        
    #     Fz = -Fz
    #     return np.array([Fx, 0.0, Fz])
    
    def force_world(self, action):

        K = action[0]
        C = action[1]
        T_gain = action[2]

        base = self.base_pos()
        ee = self.ee_pos()

        leg_vec = ee - base
        l = np.linalg.norm(leg_vec)
        leg_dir = leg_vec / l

        # --- Linear spring-damper ---
        Jp = np.zeros((3, self.m.nv))
        mj.mj_jacSite(self.m, self.d, Jp, None, self.left_tip_id)

        vel = Jp @ self.d.qvel
        ldot = leg_dir @ vel

        Fl = K*(self.ori_l - l) - C*ldot

        F_lin = Fl * leg_dir

        # --- Torsional spring ---
        alpha = np.arctan2(leg_dir[0], -leg_dir[2])
        tau_t = T_gain * (self.ori_theta - alpha)

        leg_perp = np.array([-leg_dir[2], 0.0, leg_dir[0]])
        F_tor = (tau_t / l) * leg_perp

        # --- Total world force ---
        F_world = -F_lin - F_tor
        mass = mj.mj_getTotalmass(self.m)
        weight = mass * 9.81
        # F_world[2] = F_world[2] + weight
        return F_world

    # ---------------------------------------------------
    # Torque Mapping (MuJoCo Jacobian)
    # ---------------------------------------------------

    def get_ground_contact_forces(self):

        ground_id = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, "floor")

        contact_list = []

        for i in range(self.d.ncon):
            con = self.d.contact[i]

            if con.geom1 == ground_id or con.geom2 == ground_id:
                force = np.zeros(6)
                mj.mj_contactForce(self.m, self.d, i, force)
                contact_list.append((force[:3], con))

        return contact_list

    def joint_torque(self, action):

        # Jp = np.zeros((3, self.m.nv))
        # mj.mj_jacSite(self.m, self.d, Jp, None, self.left_tip_id)
        Jp_left  = np.zeros((3, self.m.nv))
        Jp_right = np.zeros((3, self.m.nv))

        mj.mj_jacSite(self.m, self.d, Jp_left,  None, self.left_tip_id)
        mj.mj_jacSite(self.m, self.d, Jp_right, None, self.right_tip_id)

        Jp = 0.5 * (Jp_left + Jp_right)

        F = self.force_world(action)

        tau_full = Jp.T @ F

        tau_left  = tau_full[self.hip_left_dof]
        tau_right = tau_full[self.hip_right_dof]

        return np.array([tau_left, tau_right])

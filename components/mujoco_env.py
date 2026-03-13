from math import tau
import mujoco as mj
import numpy as np
from mujoco import viewer

class MuJoCoMonopedEnv:
    def __init__(self, xml_path, m, d, visualize=True):
        self.m = m
        self.d = d

        # Cache joint indices
        base_x_joint_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_x")
        base_z_joint_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_z")
        base_y_joint_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hinge_y")

        thigh_joint_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hinge1")
        knee_joint_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hinge2")

        self.base_x_joint_id = base_x_joint_id
        self.base_z_joint_id = base_z_joint_id
        self.base_y_joint_id = base_y_joint_id
        self.thigh_joint_id = thigh_joint_id
        self.knee_joint_id = knee_joint_id

        self.hip_qpos_id = m.jnt_qposadr[thigh_joint_id]
        self.knee_qpos_id = m.jnt_qposadr[knee_joint_id]
        self.hip_dof_id = m.jnt_dofadr[thigh_joint_id]
        self.knee_dof_id = m.jnt_dofadr[knee_joint_id]
        

        motor1_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, "torque1")
        motor2_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, "torque2")
        self.motor1_id = motor1_id
        self.motor2_id = motor2_id

        self.shank_geom_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "shank")
        self.root_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "root")
        self.base_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "base")
        self.prev_com_pos = None
        self.visualize = visualize  
        self.viewer = None
        if self.visualize :
            self.viewer = viewer.launch_passive(m, d)

    # -------------------------------------------------
    # State extraction (for MPC)
    # -------------------------------------------------
    def get_state(self):
        com_pos = self.d.subtree_com[self.root_body_id]
        if self.prev_com_pos is None:
            com_vel = np.zeros(3)
        else:
            com_vel = (com_pos - self.prev_com_pos) / self.m.opt.timestep
        self.prev_com_pos = com_pos.copy()

        theta = float(self.d.qpos[self.base_y_joint_id])
        thetad = float(self.d.qvel[self.base_y_joint_id])

        return np.array([
            com_pos[0],
            com_pos[2],
            theta,
            com_vel[0],
            com_vel[2],
            thetad,
            -9.81
        ], dtype=np.float64)

    def get_base_state(self):
        qpos = self.d.qpos
        qvel = self.d.qvel

        x = float(qpos[self.base_x_joint_id])
        z = float(qpos[self.base_z_joint_id])
        theta = float(qpos[self.base_y_joint_id])

        xd = float(qvel[self.base_x_joint_id])
        zd = float(qvel[self.base_z_joint_id])
        thetad = float(qvel[self.base_y_joint_id])

        return np.array([x, z, theta, xd, zd, thetad], dtype=np.float64)

    def get_joint_state(self):
        qpos = self.d.qpos
        qvel = self.d.qvel

        hip_angle = float(qpos[self.hip_qpos_id])
        knee_angle = float(qpos[self.knee_qpos_id])
        hip_vel = float(qvel[self.hip_dof_id])
        knee_vel = float(qvel[self.knee_dof_id])

        return np.array([hip_angle, knee_angle]), np.array([hip_vel, knee_vel])


    # -------------------------------------------------
    # Foot position (world frame)
    # -------------------------------------------------
    def get_foot_position(self):
        return self.get_end_effector_position()[[0, 2]]

    # -------------------------------------------------
    # End-effector position (world frame)
    # -------------------------------------------------
    def get_end_effector_position(self):
        """
        Return the world-frame position of the distal end of the shank.
        This uses the shank geom's local fromto endpoint and the link2 body pose.
        """
        if hasattr(self.m, "geom_fromto"):
            shank_fromto = self.m.geom_fromto[self.shank_geom_id]
            if np.allclose(shank_fromto, 0.0):
                p_local = self.m.geom_pos[self.shank_geom_id]
            else:
                p_local = shank_fromto[3:6]
        else:
            p_local = self.m.geom_pos[self.shank_geom_id]

        body_id = self.m.geom_bodyid[self.shank_geom_id]
        body_pos = self.d.xpos[body_id]
        body_xmat = self.d.xmat[body_id].reshape(3, 3)

        p_world = body_pos + body_xmat @ p_local
        return p_world

    # -------------------------------------------------
    # Contact detection
    # -------------------------------------------------
    def in_stance(self):
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.geom2 == self.shank_geom_id or c.geom1 == self.shank_geom_id:
                return True
        return False

    # -------------------------------------------------
    # Apply desired foot force via Jᵀ
    # -------------------------------------------------

    def foot_force_to_torque(self, foot_force):
        """
        Map Cartesian foot force to joint torques using J^T
        """
        Jp = np.zeros((3, self.m.nv))
        mj.mj_jacGeom(self.m, self.d, Jp, None, self.shank_geom_id)

        # Planar x–z force
        F = np.array([foot_force[0], 0.0, foot_force[1]])

        tau = Jp.T @ F

        # Extract actuated DOFs only
        tau_cmd = np.zeros(2)
        tau_cmd[0] = tau[self.hip_dof_id]
        tau_cmd[1] = tau[self.knee_dof_id]

        return tau_cmd

    def apply_foot_force(self, foot_force):
        """
        Apply world-frame foot force using Jacobian transpose.
        foot_force = [Fx, Fy]  (world x, z)
        """

        # Fx, Fz = foot_force

        # # Full 3D force vector (MuJoCo convention)
        # F_world = np.array([Fx, 0.0, Fz])

        # # Allocate correct-size Jacobian
        # J = np.zeros((3, self.model.nv))

        # mujoco.mj_jacGeom(
        #     self.model,
        #     self.data,
        #     J,          # jacp
        #     None,       # jacr not needed
        #     self.foot_geom
        # )

        # # Joint-space force
        # tau = J.T @ F_world

        # # Apply to qfrc_applied
        # hip_tau  = tau[self.hip_id]
        # knee_tau = tau[self.knee_id]

        # self.data.ctrl[0] = hip_tau
        # self.data.ctrl[1] = knee_tau
        tau = self.foot_force_to_torque(foot_force)
        if hasattr(self.m, "actuator_ctrlrange"):
            ctrl_min = self.m.actuator_ctrlrange[:, 0]
            ctrl_max = self.m.actuator_ctrlrange[:, 1]
            tau = np.clip(tau, ctrl_min, ctrl_max)
        self.d.ctrl[:] = tau

    def apply_joint_torques(self, tau_cmd):
        tau = np.array([tau_cmd[0], tau_cmd[1]], dtype=float)
        if hasattr(self.m, "actuator_ctrlrange"):
            ctrl_min = self.m.actuator_ctrlrange[:, 0]
            ctrl_max = self.m.actuator_ctrlrange[:, 1]
            tau = np.clip(tau, ctrl_min, ctrl_max)
        self.d.ctrl[self.motor1_id] = tau[0]
        self.d.ctrl[self.motor2_id] = tau[1]


    # -------------------------------------------------
    def step(self, n=1):
        for _ in range(n):
            mj.mj_step(self.m, self.d)
            
    
            if self.viewer is not None:
                self.viewer.sync()


import numpy as np
import mujoco as mj
import mujoco.viewer
import time
import ik_5bar as ik
import vmc_5bar as vmc_rp


# --------------------------------------------------
# Ground reaction force helper
# --------------------------------------------------

def get_grf_z(m, d):

    ground_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")

    total_grf = np.zeros(3)

    for i in range(d.ncon):

        con = d.contact[i]

        if con.geom1 == ground_id or con.geom2 == ground_id:

            force = np.zeros(6)
            mj.mj_contactForce(m, d, i, force)

            R = con.frame.reshape(3,3)

            if con.geom1 == ground_id:
                f_world = R @ force[:3]
            else:
                f_world = -R @ force[:3]

            total_grf += f_world

    return total_grf[2]


# --------------------------------------------------
# Main simulation
# --------------------------------------------------

def run(xml_path, action, ik_height, thigh_length, calf_length, hip_offset):

    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)

    # joints
    hip_left_id  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_left")
    hip_right_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_right")

    slide_z_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_z")

    hip_left_dof  = m.jnt_dofadr[hip_left_id]
    hip_right_dof = m.jnt_dofadr[hip_right_id]
    slide_z_dof   = m.jnt_dofadr[slide_z_id]

    # actuators
    hip_left_actuator  = m.actuator("motor_left").id
    hip_right_actuator = m.actuator("motor_right").id

    # --------------------------------------------------
    # Spawn pose from IK
    # --------------------------------------------------

    q1_l, q2_l, q1_r, q2_r = ik.ik_5bar(
        0.0,
        ik_height,
        thigh_length,
        calf_length,
        hip_offset
    )

    d.qpos[m.joint("hip_left").qposadr[0]] = q1_l
    d.qpos[m.joint("knee_left").qposadr[0]] = q2_l
    d.qpos[m.joint("hip_right").qposadr[0]] = q1_r
    d.qpos[m.joint("knee_right").qposadr[0]] = q2_r

    d.qpos[m.joint("slide_z").qposadr[0]] = -ik_height

    mj.mj_forward(m, d)

    controller = vmc_rp.Controller(m, d, xml_path)

    # --------------------------------------------------
    # Controller parameters
    # --------------------------------------------------

    kp = 300
    kd = 30

    phase = "LOAD"

    contact_threshold = 5
    liftoff_threshold = 2

    lean_angle = 0.2

    # --------------------------------------------------
    # Viewer loop
    # --------------------------------------------------

    with mj.viewer.launch_passive(m, d) as viewer:

        while viewer.is_running():

            hip_left_torque = 0.0
            hip_right_torque = 0.0

            # signals
            grf_z = get_grf_z(m, d)
            grf = abs(grf_z)

            base_z_vel = d.qvel[slide_z_dof]

            # --------------------------------------------------
            # Phase machine
            # --------------------------------------------------

            if phase == "LOAD":

                controller.ori_theta = 0.0

                tau = controller.joint_torque(action)

                hip_left_torque  = tau[0]
                hip_right_torque = tau[1]

                if grf > contact_threshold and base_z_vel >= 0:
                    phase = "PUSH"


            elif phase == "PUSH":

                controller.ori_theta = lean_angle

                tau = controller.joint_torque(action)

                hip_left_torque  = tau[0]
                hip_right_torque = tau[1]

                if grf < liftoff_threshold:
                    phase = "FLIGHT"


            elif phase == "FLIGHT":

                controller.ori_theta = 0.0

                q_l = d.qpos[m.joint("hip_left").qposadr[0]]
                q_r = d.qpos[m.joint("hip_right").qposadr[0]]

                qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
                qd_r = d.qvel[m.joint("hip_right").dofadr[0]]

                hip_left_torque  = kp*(q1_l - q_l) - kd*qd_l
                hip_right_torque = kp*(q1_r - q_r) - kd*qd_r

                if grf > contact_threshold:
                    phase = "RESET"


            elif phase == "RESET":

                controller.ori_theta = 0.0

                q_l = d.qpos[m.joint("hip_left").qposadr[0]]
                q_r = d.qpos[m.joint("hip_right").qposadr[0]]

                qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
                qd_r = d.qvel[m.joint("hip_right").dofadr[0]]

                hip_left_torque  = kp*(q1_l - q_l) - kd*qd_l
                hip_right_torque = kp*(q1_r - q_r) - kd*qd_r

                if base_z_vel >= 0:
                    phase = "LOAD"

            # --------------------------------------------------
            # Apply control
            # --------------------------------------------------

            d.ctrl[hip_left_actuator]  = hip_left_torque
            d.ctrl[hip_right_actuator] = hip_right_torque

            mj.mj_step(m, d)
            viewer.sync()


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar_base.xml"

    ik_height = -0.6
    thigh_length = 0.4
    calf_length = 0.4
    hip_offset = 0.05

    action = np.array([150.0, 1.0, 25.0])

    run(
        xml_path,
        action,
        ik_height,
        thigh_length,
        calf_length,
        hip_offset
    )
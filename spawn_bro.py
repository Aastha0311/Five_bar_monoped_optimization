import mujoco
import mujoco.viewer
import numpy as np
from components.ik_5bar import ik_5bar


xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar_base.xml"

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

def get_ground_reaction_force(m, d):

    total_grf = np.zeros(3)

    for i in range(d.ncon):

        contact = d.contact[i]

        geom1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        if geom1 == "floor" or geom2 == "floor":

            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)

            # first 3 components = force
            total_grf += force[:3]

    return total_grf


viewer = mujoco.viewer.launch_passive(m, d)

l1 = 0.4
l2 = 0.4
hip_offset = 0.05

# EE directly below base
x_target = 0.0
z_target = -0.7

q1_l, q2_l, q1_r, q2_r = ik_5bar(
    x_target, z_target, l1, l2, hip_offset
)

# Set joint angles
d.qpos[m.joint("hip_left").qposadr[0]] = q1_l
d.qpos[m.joint("knee_left").qposadr[0]] = q2_l
d.qpos[m.joint("hip_right").qposadr[0]] = q1_r
d.qpos[m.joint("knee_right").qposadr[0]] = q2_r

# Move root so foot touches ground
d.qpos[m.joint("slide_z").qposadr[0]] = -z_target

mujoco.mj_forward(m, d)

kp = 300
kd = 30

while viewer.is_running():

    q_l = d.qpos[m.joint("hip_left").qposadr[0]]
    q_r = d.qpos[m.joint("hip_right").qposadr[0]]

    qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
    qd_r = d.qvel[m.joint("hip_right").dofadr[0]]

    d.ctrl[0] = kp*(q1_l - q_l) - kd*qd_l
    d.ctrl[1] = kp*(q1_r - q_r) - kd*qd_r
    grf = get_ground_reaction_force(m, d)
    print("GRF:", grf)


    mujoco.mj_step(m, d)
    viewer.sync()



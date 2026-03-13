import mujoco
import mujoco.viewer
import numpy as np
import time
from components.ik_5bar import ik_5bar  # change to your file name


xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/generated_5bar.xml"

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

viewer = mujoco.viewer.launch_passive(m, d)


# Geometry
l1 = 0.4
l2 = 0.4
hip_offset = 0.05

# Desired foot position (centered under body)
x_target = 0.0
z_target = -0.6

# Compute IK
q1_l, q2_l, q1_r, q2_r = ik_5bar(
    x_target, z_target, l1, l2, hip_offset
)

# Set joint angles
d.qpos[m.joint("j1_left").qposadr[0]] = q1_l
d.qpos[m.joint("j2_left").qposadr[0]] = q2_l
d.qpos[m.joint("j1_right").qposadr[0]] = q1_r
d.qpos[m.joint("j2_right").qposadr[0]] = q2_r

#mujoco.mj_forward(m, d)

# Adjust root height so foot touches ground
left_site = m.site("left_tip").id
foot_z = d.site_xpos[left_site][2]

slide_z_id = m.joint("slide_z").qposadr[0]
d.qpos[slide_z_id] -= foot_z

#mujoco.mj_forward(m, d)


# PD gains
kp = 200
kd = 20

while viewer.is_running():

    q1_left = d.qpos[m.joint("j1_left").qposadr[0]]
    q1_right = d.qpos[m.joint("j1_right").qposadr[0]]

    qd1_left = d.qvel[m.joint("j1_left").dofadr[0]]
    qd1_right = d.qvel[m.joint("j1_right").dofadr[0]]

    tau_left = kp * (q1_l - q1_left) - kd * qd1_left
    tau_right = kp * (q1_r - q1_right) - kd * qd1_right

    d.ctrl[0] = tau_left
    d.ctrl[1] = tau_right

    mujoco.mj_step(m, d)
    viewer.sync()



import time
import sys
import numpy as np
import mujoco
from mujoco import viewer



def main():
    xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar_simple.xml"
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)

    joint1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "1")
    joint4_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "4")
    dof1 = m.jnt_dofadr[joint1_id]
    dof4 = m.jnt_dofadr[joint4_id]

    kp = 20.0
    kd = 1.0
    q_des = np.array([0.0, 0.0])

    with viewer.launch_passive(m, d) as mj_viewer:
        while mj_viewer.is_running():
            step_start = time.time()

            q = np.array([d.qpos[m.jnt_qposadr[joint1_id]], d.qpos[m.jnt_qposadr[joint4_id]]])
            qd = np.array([d.qvel[dof1], d.qvel[dof4]])
            tau = kp * (q_des - q) + kd * (0.0 - qd)

            d.qfrc_applied[dof1] = tau[0]
            d.qfrc_applied[dof4] = tau[1]

            mujoco.mj_step(m, d)
            mj_viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

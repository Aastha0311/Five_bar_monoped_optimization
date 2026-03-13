import time
import numpy as np
import mujoco
from mujoco import viewer


def main():
    xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/diamond_simple.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    joint_left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hip_left")
    joint_right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hip_right")
    dof_left = m.jnt_dofadr[joint_left]
    dof_right = m.jnt_dofadr[joint_right]

    kp = 20.0
    kd = 1.0
    q_des = np.array([0.0, 0.0])

    with viewer.launch_passive(m, d) as mj_viewer:
        while mj_viewer.is_running():
            step_start = time.time()

            q = np.array([
                d.qpos[m.jnt_qposadr[joint_left]],
                d.qpos[m.jnt_qposadr[joint_right]],
            ])
            qd = np.array([d.qvel[dof_left], d.qvel[dof_right]])
            tau = kp * (q_des - q) + kd * (0.0 - qd)

            d.qfrc_applied[dof_left] = tau[0]
            d.qfrc_applied[dof_right] = tau[1]

            mujoco.mj_step(m, d)
            mj_viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

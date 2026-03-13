import time
import numpy as np
import mujoco
from mujoco import viewer


def main():
    xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/planar_test_5bar2width.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    act1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "torque1")
    act2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "torque2")

    j1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hinge1")
    j2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hinge2")

    q_hold = None
    kp = 12.0
    kd = 0.6

    with viewer.launch_passive(m, d) as mj_viewer:
        while mj_viewer.is_running():
            step_start = time.time()

            if q_hold is None:
                q_hold = np.array([
                    d.qpos[m.jnt_qposadr[j1]],
                    d.qpos[m.jnt_qposadr[j2]],
                ])

            q1 = d.qpos[m.jnt_qposadr[j1]]
            q2 = d.qpos[m.jnt_qposadr[j2]]
            v1 = d.qvel[m.jnt_dofadr[j1]]
            v2 = d.qvel[m.jnt_dofadr[j2]]

            tau1 = kp * (q_hold[0] - q1) - kd * v1
            tau2 = kp * (q_hold[1] - q2) - kd * v2

            a1_min, a1_max = m.actuator_ctrlrange[act1]
            a2_min, a2_max = m.actuator_ctrlrange[act2]
            d.ctrl[act1] = np.clip(tau1, a1_min, a1_max)
            d.ctrl[act2] = np.clip(tau2, a2_min, a2_max)

            mujoco.mj_step(m, d)
            mj_viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

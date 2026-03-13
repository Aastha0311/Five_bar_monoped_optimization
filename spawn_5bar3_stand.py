import time
import numpy as np
import mujoco
from mujoco import viewer


def main():
    xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar3.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    left_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_torque_left")
    right_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_torque_right")

    j1_left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "j1_left")
    j1_right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "j1_right")

    q_hold = None
    kp_hold = 12.0
    kd_hold = 0.6

    with viewer.launch_passive(m, d) as mj_viewer:
        while mj_viewer.is_running():
            step_start = time.time()

            if q_hold is None:
                q_hold = np.array([
                    d.qpos[m.jnt_qposadr[j1_left]],
                    d.qpos[m.jnt_qposadr[j1_right]],
                ])

            q_left = d.qpos[m.jnt_qposadr[j1_left]]
            q_right = d.qpos[m.jnt_qposadr[j1_right]]
            v_left = d.qvel[m.jnt_dofadr[j1_left]]
            v_right = d.qvel[m.jnt_dofadr[j1_right]]

            tau_left = kp_hold * (q_hold[0] - q_left) - kd_hold * v_left
            tau_right = kp_hold * (q_hold[1] - q_right) - kd_hold * v_right

            left_min, left_max = m.actuator_ctrlrange[left_act]
            right_min, right_max = m.actuator_ctrlrange[right_act]
            d.ctrl[left_act] = np.clip(tau_left, left_min, left_max)
            d.ctrl[right_act] = np.clip(tau_right, right_min, right_max)

            mujoco.mj_step(m, d)
            mj_viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

import time
import numpy as np
import mujoco
from mujoco import viewer


def main():
    xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar3_centered.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    left_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_torque_left")
    right_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_torque_right")

    floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    contact_geoms = []
    for name in ["l2_left_geom", "l2_right_geom"]:
        try:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                contact_geoms.append(gid)
        except Exception:
            pass

    in_contact = False

    j1_left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "j1_left")
    j1_right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "j1_right")

    q_hold = None
    kp = 12.0
    kd = 0.6

    total_mass = float(np.sum(m.body_mass))
    print(f"total_mass={total_mass:.4f}")

    step_count = 0

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

            tau_left = kp * (q_hold[0] - q_left) - kd * v_left
            tau_right = kp * (q_hold[1] - q_right) - kd * v_right

            left_min, left_max = m.actuator_ctrlrange[left_act]
            right_min, right_max = m.actuator_ctrlrange[right_act]
            d.ctrl[left_act] = np.clip(tau_left, left_min, left_max)
            d.ctrl[right_act] = np.clip(tau_right, right_min, right_max)

            contact_now = False
            left_force = 0.0
            right_force = 0.0

            if contact_geoms:
                for i in range(d.ncon):
                    c = d.contact[i]
                    if (c.geom1 in contact_geoms and c.geom2 == floor_geom) or (
                        c.geom2 in contact_geoms and c.geom1 == floor_geom
                    ):
                        cf = np.zeros(6)
                        mujoco.mj_contactForce(m, d, i, cf)
                        normal_f = float(cf[0])
                        if c.geom1 == contact_geoms[0] or c.geom2 == contact_geoms[0]:
                            left_force += normal_f
                        if len(contact_geoms) > 1 and (
                            c.geom1 == contact_geoms[1] or c.geom2 == contact_geoms[1]
                        ):
                            right_force += normal_f
                        contact_now = contact_now or (normal_f > 1e-3)

            if contact_now and not in_contact:
                print(
                    f"contact at t={d.time:.3f} leftF={left_force:.3f} rightF={right_force:.3f}"
                )
            in_contact = contact_now

            if step_count % 200 == 0 and in_contact:
                print(
                    f"t={d.time:.3f} leftF={left_force:.3f} rightF={right_force:.3f}"
                )
            step_count += 1

            mujoco.mj_step(m, d)
            mj_viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

import time
import numpy as np
import mujoco
from mujoco import viewer


def main():
    xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/planar_test_5bar2width.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)


    left_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_torque_left")
    right_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_torque_right")

    floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    contact_geoms = []

    for name in ["pen_tip", "l2_left_geom", "l2_right_geom"]:
        try:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                contact_geoms.append(gid)
        except Exception:
            pass

    in_contact = False
    step_count = 0

    hold_init = True
    q_hold = None
    kp_hold = 10.0
    kd_hold = 0.4

    # GRF-based stance control (maps desired GRF to joint torques)
    v_des = 0.01
    kv_forward = 200.0
    max_fx = 120.0
    kp_z = 1200.0
    kd_z = 60.0

    left_touch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "touch_left")
    right_touch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "touch_right")

    com_prev = None
    com_xdot = 0.0

    with viewer.launch_passive(m, d) as mj_viewer:
        while mj_viewer.is_running():
            step_start = time.time()

            if hold_init:
                if q_hold is None:
                    j1_left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "j1_left")
                    j1_right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "j1_right")
                    slide_x = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")
                    slide_z = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "slide_z")
                    site_pen = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "site_pen")
                    left_dof = m.jnt_dofadr[j1_left]
                    right_dof = m.jnt_dofadr[j1_right]
                    z_hold = d.qpos[m.jnt_qposadr[slide_z]]
                    total_mass = float(np.sum(m.body_mass))
                    q_hold = np.array([
                        d.qpos[m.jnt_qposadr[j1_left]],
                        d.qpos[m.jnt_qposadr[j1_right]],
                    ])

                com_now = d.subtree_com[0].copy()
                if com_prev is None:
                    com_prev = com_now
                com_xdot = (com_now[0] - com_prev[0]) / m.opt.timestep
                com_prev = com_now

                q_left = d.qpos[m.jnt_qposadr[j1_left]]
                q_right = d.qpos[m.jnt_qposadr[j1_right]]
                v_left = d.qvel[m.jnt_dofadr[j1_left]]
                v_right = d.qvel[m.jnt_dofadr[j1_right]]

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
                            if c.geom1 == contact_geoms[1] or c.geom2 == contact_geoms[1]:
                                right_force += normal_f
                            contact_now = contact_now or (normal_f > 1e-3)

                if contact_now != in_contact:
                    in_contact = contact_now
                    print(f"contact: {in_contact} at t={d.time:.3f}")
                    if floor_geom >= 0:
                        if in_contact:
                            m.geom_rgba[floor_geom] = np.array([0.8, 0.2, 0.2, 1.0])
                        else:
                            m.geom_rgba[floor_geom] = np.array([0.2, 0.2, 0.2, 1.0])

                xdot = d.qvel[m.jnt_dofadr[slide_x]]
                z = d.qpos[m.jnt_qposadr[slide_z]]
                zdot = d.qvel[m.jnt_dofadr[slide_z]]

                if in_contact:
                    fx_des = np.clip(kv_forward * (v_des - com_xdot), -max_fx, max_fx)
                    fz_des = total_mass * 9.81 + kp_z * (z_hold - z) - kd_z * zdot
                    fz_des = np.clip(fz_des, 0.0, 3.0 * total_mass * 9.81)

                    jacp = np.zeros((3, m.nv))
                    mujoco.mj_jacSite(m, d, jacp, None, site_pen)
                    f_world = np.array([fx_des, 0.0, fz_des])

                    tau_left = float(jacp[:, left_dof].dot(f_world))
                    tau_right = float(jacp[:, right_dof].dot(f_world))
                else:
                    tau_left = 0.0
                    tau_right = 0.0

                # Clamp to actuator limits
                left_min, left_max = m.actuator_ctrlrange[left_act]
                right_min, right_max = m.actuator_ctrlrange[right_act]
                d.ctrl[left_act] = np.clip(tau_left, left_min, left_max)
                d.ctrl[right_act] = np.clip(tau_right, right_min, right_max)

                if step_count % 200 == 0:
                    print(
                        f"t={d.time:.3f} contact={in_contact} "
                        f"leftF={left_force:.3f} rightF={right_force:.3f} "
                        f"xdot={xdot:.3f} com_xdot={com_xdot:.3f} v_des={v_des:.3f} "
                        f"fx_des={fx_des:.1f} fz_des={fz_des:.1f}"
                    )
                    if d.ncon > 0:
                        for i in range(min(d.ncon, 5)):
                            c = d.contact[i]
                            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                            g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                            print(f"  contact: {g1} <-> {g2}")
                step_count += 1

            mujoco.mj_step(m, d)
            mj_viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

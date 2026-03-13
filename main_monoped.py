import numpy as np
import mujoco
from components.mujoco_env import MuJoCoMonopedEnv
from components.controller_monoped import Controller
from components.robot_monoped import Robot_monoped_2D


def _geom_length_fromto(m, geom_id):
    if hasattr(m, "geom_fromto"):
        fromto = m.geom_fromto[geom_id]
        if not np.allclose(fromto, 0.0):
            return float(np.linalg.norm(fromto[3:6] - fromto[0:3]))

    size = m.geom_size[geom_id]
    if size.shape[0] >= 2 and size[1] > 0:
        return float(2.0 * size[1])

    return 0.0


def extract_robot_params_from_xml(m):
    thigh_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "thigh")
    shank_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "shank")
    thigh_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link1")
    shank_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link2")

    exclude_bodies = {"world", "root"}
    body_names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)
    ]
    body_mask = [name not in exclude_bodies for name in body_names]

    total_mass = float(np.sum(m.body_mass[body_mask]))
    moment_of_inertia = float(np.sum(m.body_inertia[body_mask, 1]))
    # nv = m.nv
    # M = np.zeros((nv, nv))
    # mujoco.mj_fullM(m, M, d.qM)

    # hinge_id = m.joint("hinge_y").dofadr[0]
    # I_effective = M[hinge_id, hinge_id]
    #print(I_effective)
    L_thigh = _geom_length_fromto(m, thigh_geom_id)
    L_shank = _geom_length_fromto(m, shank_geom_id)

    if hasattr(m, "geom_mass"):
        m_thigh = float(m.geom_mass[thigh_geom_id])
        m_shank = float(m.geom_mass[shank_geom_id])
    else:
        m_thigh = float(m.body_mass[thigh_body_id])
        m_shank = float(m.body_mass[shank_body_id])

    friction_coeff = float(m.geom_friction[shank_geom_id][0])

    density = 0.0
    if hasattr(m, "geom_density"):
        density = float(m.geom_density[thigh_geom_id])

    return {
        "mass": total_mass,
        "moment_of_inertia": moment_of_inertia,
        "L_thigh": L_thigh,
        "L_shank": L_shank,
        "friction_coeff": friction_coeff,
        "density": density,
        "m_thigh": m_thigh,
        "m_shank": m_shank,
    }


if __name__ == "__main__":
    # -----------------------------
    # Load MuJoCo model
    # -----------------------------
    xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/planar_test.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # -----------------------------
    # Extract physical parameters and build robot model
    # -----------------------------
    robot_params = extract_robot_params_from_xml(m)
    
    robot = Robot_monoped_2D(**robot_params)

    # -----------------------------
    # Controller and environment setup
    # -----------------------------
    controller = Controller(
        robot,
        time_horizon=10,
        stepping_frequency=2.0,
        control_dt=0.01,
        raibert_gain=0.08,
    )
    env = MuJoCoMonopedEnv(xml_path, m, d, visualize=True)

    # Use the current MuJoCo COM height as the initial nominal height
    robot.nominal_com_height = float(env.get_state()[1])

    # -----------------------------
    # Desired motion
    # -----------------------------
    desired_velocity = np.array([0.5, 0.0])
    desired_ang_vel = 0.0

    # -----------------------------
    # Control loop
    # -----------------------------
    t = 0.0
    steps_per_control = max(1, int(controller.control_dt / m.opt.timestep))
    #steps_per_control = 1
    log_stride = max(1, int(0.1 / controller.control_dt))
    step_count = 0
    prev_foot_force = np.array([0.0, 0.0])
    force_alpha = 0.2
    swing_time = 0.0
    swing_duration = 0.5 / controller.stepping_frequency

    # Initialize stance from current contact/height
    end_effector_pos = env.get_end_effector_position()[[0, 2]]
    last_in_stance = env.in_stance() or (end_effector_pos[1] <= robot.ground_level + 0.02)

    while env.viewer is None or env.viewer.is_running():
        # Feedback from simulator
        state = env.get_state()
        base_state = env.get_base_state()
        end_effector_pos = env.get_end_effector_position()[[0, 2]]
        contact_by_height = end_effector_pos[1] <= (robot.ground_level + 0.02)
        in_stance = env.in_stance() or contact_by_height

        base_pos = base_state[0:3]
        joint_pos, joint_vel = env.get_joint_state()

        # Plan control action
        if in_stance:
            if not last_in_stance:
                swing_time = 0.0
            if t < 0.5:
                foot_force, _ = controller.stand(
                    state,
                    end_effector_pos,
                    robot.nominal_com_height,
                )
            else:
                foot_force, _ = controller.walk(
                    state,
                    end_effector_pos,
                    desired_velocity,
                    desired_ang_vel,
                    in_stance,
                )

            # Apply stance GRF to MuJoCo
            foot_force = (1.0 - force_alpha) * prev_foot_force + force_alpha * foot_force
            prev_foot_force = foot_force
            env.apply_foot_force(foot_force)
        else:
            if last_in_stance:
                swing_time = 0.0
            # Swing control: track desired foot position with joint PD
            T = 1.0 / controller.stepping_frequency
            desired_foot_pos = np.array([
                base_state[0] + 0.5 * base_state[3] * T
                + controller.Kp * (desired_velocity[0] - base_state[3]),
                robot.ground_level
            ])

            swing_time += controller.control_dt
            phase = min(1.0, swing_time / swing_duration)
            desired_foot_pos[1] = (
                robot.ground_level
                + controller.swing_height * np.sin(np.pi * phase)
            )
            if phase >= 1.0:
                desired_foot_pos[1] = robot.ground_level - 0.05

            hip_des, knee_des = robot.inverse_kinematics(base_pos, desired_foot_pos)
            q_des = np.array([hip_des, knee_des])

            kp = np.array([80.0, 80.0])
            kd = np.array([5.0, 5.0])
            tau_cmd = kp * (q_des - joint_pos) + kd * (0.0 - joint_vel)
            tau_g = robot.calculate_gravitational_torques(joint_pos[0], joint_pos[1])
            tau_cmd = tau_cmd + np.array(tau_g)
            tau_cmd = np.clip(tau_cmd, -120.0, 120.0)

            env.apply_joint_torques(tau_cmd)
        env.step(n=steps_per_control)

        last_in_stance = in_stance

        t += controller.control_dt
        step_count += 1

        if step_count % log_stride == 0:
            xdot = state[3]
            zdot = state[4]
            ee_z = end_effector_pos[1]
            print(f"t={t:6.2f}  xdot={xdot:7.3f}  zdot={zdot:7.3f}  ee_z={ee_z:6.3f}  stance={in_stance}")



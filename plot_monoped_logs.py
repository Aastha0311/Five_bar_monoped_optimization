import numpy as np
import mujoco
import matplotlib.pyplot as plt

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


def run_and_log(sim_time=10.0):
    xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/planar_test.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    robot_params = extract_robot_params_from_xml(m)
    robot = Robot_monoped_2D(**robot_params)

    controller = Controller(
        robot,
        time_horizon=10,
        stepping_frequency=2.0,
        control_dt=0.01,
        raibert_gain=0.08,
    )
    env = MuJoCoMonopedEnv(xml_path, m, d, visualize=False)

    robot.nominal_com_height = float(env.get_state()[1])

    desired_velocity = np.array([0.5, 0.0])
    desired_ang_vel = 0.0

    steps_per_control = max(1, int(controller.control_dt / m.opt.timestep))
    #steps_per_control = 5
    root_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "root")

    t = 0.0
    swing_time = 0.0
    swing_duration = 0.5 / controller.stepping_frequency

    logs = {
        "t": [],
        "hip_angle": [],
        "knee_angle": [],
        "hip_vel": [],
        "knee_vel": [],
        "ee_x": [],
        "ee_z": [],
        "tau_hip": [],
        "tau_knee": [],
        "base_x": [],
        "base_z": [],
        "base_xdot": [],
        "base_zdot": [],
        "com_x": [],
        "com_z": [],
        "com_xdot": [],
        "com_zdot": [],
        "stance": [],
    }

    while t < sim_time:
        state = env.get_state()
        base_state = env.get_base_state()
        end_effector_pos = env.get_end_effector_position()
        contact_by_height = end_effector_pos[2] <= (robot.ground_level + 0.02)
        in_stance = env.in_stance() or contact_by_height

        com_pos = d.subtree_com[root_body_id]
        com_vel = np.array([state[3], 0.0, state[4]])

        joint_pos, joint_vel = env.get_joint_state()

        if in_stance:
            swing_time = 0.0
            foot_force, _ = controller.walk(
                state,
                end_effector_pos[[0, 2]],
                desired_velocity,
                desired_ang_vel,
                in_stance,
            )
            env.apply_foot_force(foot_force)
        else:
            _, desired_foot_pos = controller.walk(
                state,
                end_effector_pos[[0, 2]],
                desired_velocity,
                desired_ang_vel,
                in_stance,
            )
            swing_time += controller.control_dt
            phase = min(1.0, swing_time / swing_duration)
            desired_foot_pos[1] = (
                robot.ground_level
                + controller.swing_height * np.sin(np.pi * phase)
            )
            if phase >= 1.0:
                desired_foot_pos[1] = robot.ground_level - 0.05

            hip_des, knee_des = robot.inverse_kinematics(state[0:3], desired_foot_pos)
            q_des = np.array([hip_des, knee_des])

            kp = np.array([80.0, 80.0])
            kd = np.array([5.0, 5.0])
            tau_cmd = kp * (q_des - joint_pos) + kd * (0.0 - joint_vel)
            tau_g = robot.calculate_gravitational_torques(joint_pos[0], joint_pos[1])
            tau_cmd = tau_cmd + np.array(tau_g)
            tau_cmd = np.clip(tau_cmd, -120.0, 120.0)
            env.apply_joint_torques(tau_cmd)

        env.step(n=steps_per_control)

        logs["t"].append(t)
        logs["hip_angle"].append(joint_pos[0])
        logs["knee_angle"].append(joint_pos[1])
        logs["hip_vel"].append(joint_vel[0])
        logs["knee_vel"].append(joint_vel[1])
        logs["ee_x"].append(end_effector_pos[0])
        logs["ee_z"].append(end_effector_pos[2])
        logs["tau_hip"].append(d.actuator_force[env.motor1_id])
        logs["tau_knee"].append(d.actuator_force[env.motor2_id])
        logs["base_x"].append(base_state[0])
        logs["base_z"].append(base_state[1])
        logs["base_xdot"].append(base_state[3])
        logs["base_zdot"].append(base_state[4])
        logs["com_x"].append(com_pos[0])
        logs["com_z"].append(com_pos[2])
        logs["com_xdot"].append(com_vel[0])
        logs["com_zdot"].append(com_vel[2])
        logs["stance"].append(float(in_stance))

        t += controller.control_dt

    return logs


def plot_logs(logs):
    t = np.array(logs["t"])

    fig, axs = plt.subplots(4, 2, figsize=(12, 10), sharex=True)

    axs[0, 0].plot(t, logs["hip_angle"], label="hip")
    axs[0, 0].plot(t, logs["knee_angle"], label="knee")
    axs[0, 0].set_ylabel("joint angle (rad)")
    axs[0, 0].legend()

    axs[0, 1].plot(t, logs["hip_vel"], label="hip")
    axs[0, 1].plot(t, logs["knee_vel"], label="knee")
    axs[0, 1].set_ylabel("joint vel (rad/s)")
    axs[0, 1].legend()

    axs[1, 0].plot(t, logs["ee_x"], label="x")
    axs[1, 0].plot(t, logs["ee_z"], label="z")
    axs[1, 0].set_ylabel("end effector (m)")
    axs[1, 0].legend()

    axs[1, 1].plot(t, logs["tau_hip"], label="hip")
    axs[1, 1].plot(t, logs["tau_knee"], label="knee")
    axs[1, 1].set_ylabel("torque (Nm)")
    axs[1, 1].legend()

    axs[2, 0].plot(t, logs["base_x"], label="x")
    axs[2, 0].plot(t, logs["base_z"], label="z")
    axs[2, 0].set_ylabel("base pos (m)")
    axs[2, 0].legend()

    axs[2, 1].plot(t, logs["base_xdot"], label="x")
    axs[2, 1].plot(t, logs["base_zdot"], label="z")
    axs[2, 1].set_ylabel("base vel (m/s)")
    axs[2, 1].legend()

    axs[3, 0].plot(t, logs["com_x"], label="x")
    axs[3, 0].plot(t, logs["com_z"], label="z")
    axs[3, 0].set_ylabel("COM pos (m)")
    axs[3, 0].legend()

    axs[3, 1].plot(t, logs["com_xdot"], label="x")
    axs[3, 1].plot(t, logs["com_zdot"], label="z")
    axs[3, 1].set_ylabel("COM vel (m/s)")
    axs[3, 1].legend()

    axs[3, 0].set_xlabel("time (s)")
    axs[3, 1].set_xlabel("time (s)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logs = run_and_log(sim_time=10.0)
    plot_logs(logs)

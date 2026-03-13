import mujoco as mj
import mujoco.viewer
import numpy as np
import time
import ik_5bar as ik_5bar
import vmc_5bar as vmc_rp


# ================================================================
# USER SETTINGS
# ================================================================

XML_PATH = "/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar_base.xml"   # <-- your base xml
hip_offset = 0.05               # <-- match your XML hip offset
l1 = 0.4
l2 = 0.4
ik_height = -0.6
hip_peak_torque = 80
action = np.array([40, 0.5, 5])  # spring_k, damper_k, torsion_k


# ================================================================
# LOAD MODEL
# ================================================================

m = mj.MjModel.from_xml_path(XML_PATH)
d = mj.MjData(m)

# Joint IDs
hip_left_id  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_left")
hip_right_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_right")

hip_left_dof  = m.jnt_dofadr[hip_left_id]
hip_right_dof = m.jnt_dofadr[hip_right_id]

slide_x_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_x")
slide_z_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_z")

slide_x_dof = m.jnt_dofadr[slide_x_id]
slide_z_dof = m.jnt_dofadr[slide_z_id]

# Actuators
hip_left_actuator  = m.actuator('motor_left').id
hip_right_actuator = m.actuator('motor_right').id


# ================================================================
# IK SPAWN
# ================================================================

theta1, theta2 = ik.inverse_kinematics(
    0,
    ik_height,
    l1,
    l2,
    hip_offset
)

d.qpos[hip_left_dof]  = theta1
d.qpos[hip_right_dof] = theta1

mj.mj_forward(m, d)


# ================================================================
# JUMP TRACKING
# ================================================================

jump_results = []
jump_count = 0
jump_started = False
jump_phase = "ground"

current_jump_max_height = -1e12
current_jump_start_x = 0

start_time = time.time()
kp = 300
kd = 30


# ================================================================
# VIEWER LOOP
# ================================================================

with mj.viewer.launch_passive(m, d) as viewer:

    while viewer.is_running() and jump_count < 3:

        t_elapsed = time.time() - start_time

        # ===================================================
        # 5 SECOND STABILIZATION
        # ===================================================
        if t_elapsed < 5.0:

            hip_left_error  = theta1 - d.qpos[hip_left_dof]
            hip_right_error = theta1 - d.qpos[hip_right_dof]

            hip_left_torque  = 200 * hip_left_error  - 20 * d.qvel[hip_left_dof]
            hip_right_torque = 200 * hip_right_error - 20 * d.qvel[hip_right_dof]

            d.ctrl[hip_left_actuator]  = hip_left_torque
            d.ctrl[hip_right_actuator] = hip_right_torque

            mj.mj_step(m, d)
            viewer.sync()
            continue

        # ===================================================
        # CONTROLLER AFTER STABILIZATION
        # ===================================================

        controller = vmc_rp.Controller(m, d, XML_PATH, theta1, theta2)

        contact_force = controller.get_ground_contact_forces()
        forces = np.array([0,0,0])

        if len(contact_force) > 0:
            forces = contact_force[0][0]

        on_ground = not np.all(forces == 0)
        base_vel_z = d.qvel[slide_z_dof]

        # -----------------------
        # Jump detection
        # -----------------------
        if on_ground:

            if not jump_started and base_vel_z > 0:
                jump_started = True
                jump_phase = "ground_contact"
                current_jump_max_height = d.qpos[slide_z_dof]
                current_jump_start_x = d.qpos[slide_x_dof]

            elif jump_started and jump_phase == "air" and base_vel_z < 0:
                jump_phase = "landing"

            elif jump_started and jump_phase == "landing" and base_vel_z >= 0:
                jump_count += 1

                jump_x_end = d.qpos[slide_x_dof]
                height = current_jump_max_height
                euclidean_value = np.sqrt(height**2 + jump_x_end**2)

                jump_results.append((height, jump_x_end, euclidean_value))

                print(f"\nJump {jump_count}")
                print("Max height:", height)
                print("Jump X end:", jump_x_end)
                print("Euclidean:", euclidean_value)

                jump_started = False
                jump_phase = "ground"

        else:
            if jump_started and jump_phase == "ground_contact":
                jump_phase = "air"

            if jump_started and jump_phase == "air":
                current_jump_max_height = max(
                    current_jump_max_height,
                    d.qpos[slide_z_dof]
                )

        # -----------------------
        # Apply controller torques
        # -----------------------
        if on_ground:
            tau = controller.joint_torque(action)
            hip_left_torque  = np.clip(tau[0], -hip_peak_torque, hip_peak_torque)
            hip_right_torque = np.clip(tau[1], -hip_peak_torque, hip_peak_torque)
        else:
            hip_left_error  = theta1 - d.qpos[hip_left_dof]
            hip_right_error = theta1 - d.qpos[hip_right_dof]

            hip_left_torque  = 100 * hip_left_error  - 10 * d.qvel[hip_left_dof]
            hip_right_torque = 100 * hip_right_error - 10 * d.qvel[hip_right_dof]

        d.ctrl[hip_left_actuator]  = hip_left_torque
        d.ctrl[hip_right_actuator] = hip_right_torque

        mj.mj_step(m, d)
        viewer.sync()

    print("\nAll jumps complete.")
    print("Results:", jump_results)
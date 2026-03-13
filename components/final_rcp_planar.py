import numpy as np
import mujoco as mj
import time
import sys, os
import ik_5bar as ik
import vmc_5bar as vmc_rp
# import ik as ik
import random
import mujoco.viewer
# import vmc_roc_angle as vmc_ra_angle
import pandas as pd
from scipy.interpolate import interp1d

# Set fixed random seed for deterministic behavior
np.random.seed(0)
random.seed(0)

def run(xml_path, action, ik_value, hip_peak_torque, thigh_length, calf_length, hip_offset):

    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)

    m.opt.iterations = 500
    m.opt.tolerance = 1e-10

    # -----------------------
    # Joint IDs
    # -----------------------
    hip_left_id  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_left")
    hip_right_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_right")

    hip_left_dof  = m.jnt_dofadr[hip_left_id]
    hip_right_dof = m.jnt_dofadr[hip_right_id]

    slide_x_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_x")
    slide_z_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_z")

    slide_x_dof = m.jnt_dofadr[slide_x_id]
    slide_z_dof = m.jnt_dofadr[slide_z_id]

    # -----------------------
    # Actuators
    # -----------------------
    hip_left_actuator_id  = m.actuator('motor_left').id
    hip_right_actuator_id = m.actuator('motor_right').id

    # -----------------------
    # Spawn from IK
    # -----------------------
    q1_l, q2_l, q1_r, q2_r = ik.ik_5bar(0.0, ik_value, thigh_length, calf_length, hip_offset)
    d.qpos[m.joint("hip_left").qposadr[0]] = q1_l
    d.qpos[m.joint("knee_left").qposadr[0]] = q2_l
    d.qpos[m.joint("hip_right").qposadr[0]] = q1_r
    d.qpos[m.joint("knee_right").qposadr[0]] = q2_r

    # Move root so foot touches ground
    d.qpos[m.joint("slide_z").qposadr[0]] = -ik_value
    mujoco.mj_forward(m, d)

    # d.qpos[hip_left_dof]  = q1_l
    # d.qpos[hip_right_dof] = q1_r

    # -----------------------
    # Jump tracking
    # -----------------------
    jump_results = []
    jump_count = 0

    jump_started = False
    jump_phase = "ground"

    current_jump_max_height = -1e12
    current_jump_start_x = 0

    start = time.time()
    kp = 300
    kd = 30
    controller = vmc_rp.Controller(m, d, xml_path)
    phase = "LOAD"      # LOAD → PUSH → FLIGHT → RESET
    push_time = 0.0
    push_duration = 0.08   # 80 ms push window
    lean_angle = 0.2       # forward lean only during push
    with mujoco.viewer.launch_passive(m, d) as viewer:
        #while time.time() - start < 25 and jump_count < 3:
        while viewer.is_running():
        #while viewer.is_running() and jump_count < 3:

            t_elapsed = time.time() - start

            # ===================================================
            # 5 SECOND IK STABILIZATION (ADDED BLOCK)
            # ===================================================
            if t_elapsed < 2.0:

                # hip_left_error  = q1_l - d.qpos[hip_left_dof]
                # hip_right_error = q1_r - d.qpos[hip_right_dof]

                # hip_left_torque  = 200 * hip_left_error  - 20 * d.qvel[hip_left_dof]
                # hip_right_torque = 200 * hip_right_error - 20 * d.qvel[hip_right_dof]
                q_l = d.qpos[m.joint("hip_left").qposadr[0]]
                q_r = d.qpos[m.joint("hip_right").qposadr[0]]

                qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
                qd_r = d.qvel[m.joint("hip_right").dofadr[0]]

                d.ctrl[0] = kp*(q1_l - q_l) - kd*qd_l
                d.ctrl[1] = kp*(q1_r - q_r) - kd*qd_r

                # hip_left_torque  = np.clip(hip_left_torque,  -hip_peak_torque, hip_peak_torque)
                # hip_right_torque = np.clip(hip_right_torque, -hip_peak_torque, hip_peak_torque)

                # d.ctrl[hip_left_actuator_id]  = hip_left_torque
                # d.ctrl[hip_right_actuator_id] = hip_right_torque

                # print("Motor torques:")
                # print("  left :", kp*(q1_l - q_l) - kd*qd_l)
                # print("  right:", kp*(q1_r - q_r) - kd*qd_r)

                # # --- Joint reaction torques (actual) ---
                # print("Joint qfrc_actuator:")
                # print("  left :", d.qfrc_actuator[hip_left_dof])
                # print("  right:", d.qfrc_actuator[hip_right_dof])

                # --- Ground Reaction Forces from MuJoCo ---
                grf = compute_ground_reaction_force(m, d)

                print("GRF world:", grf)
                # print("GRF vertical:", grf[2])
                # print("mg:", mj.mj_getTotalmass(m)*9.81)
                # print("base z accel:", d.qacc[slide_z_dof])

                

                # # --- Base acceleration ---
                # # print("Base z accel:", d.qacc[slide_z_dof])
                # # print("Base z vel  :", d.qvel[slide_z_dof])

                # print("=================================================")
                # for i in range(d.ncon):
                #     con = d.contact[i]
                #     print("Contact normal:", con.frame.reshape(3,3)[:,2])
                for i in range(d.ncon):
                    con = d.contact[i]
                    g1 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, con.geom1)
                    g2 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, con.geom2)

                    R = con.frame.reshape(3,3)
                    normal = R[:,2]

                    print("Contact:", g1, "<->", g2)
                    print("Normal:", normal)
                    print("----")

                floor_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")
                print("Floor geom quat:", m.geom_quat[floor_id])
                print("World z axis:", m.stat.extent)
                print("Gravity:", m.opt.gravity)
                mj.mj_step(m, d)
                viewer.sync()
                
                continue
            # ===================================================

            
            contact_force = controller.get_ground_contact_forces()

            forces = np.array([0,0,0])
            if len(contact_force) > 0:
                forces = contact_force[0][0]
            
            # print("Initial distance", controller.distance())
            # print ("ori_l", controller. ori_l)
            # ---------------- DEBUG BLOCK ----------------
            alpha = controller.equivalent_orientation()
            F_world = controller.force_world(action)
            Fl = controller.total_linear_force(action)

            mass = mj.mj_getTotalmass(m)
            weight = mass * 9.81

            

            on_ground = not np.all(forces == 0)
            #on_ground = (d.ncon > 0)
            base_vel_z = d.qvel[slide_z_dof]

            # -----------------------
            # Jump detection
            # -----------------------
            # print("------ DEBUG ------")
            # print("alpha (deg):", np.degrees(alpha))
            # print("Fl:", Fl)
            # print("Fx:", F_world[0])
            # print("Fz:", F_world[2])
            # print("mg:", weight)
            # # print("Net vertical (Fz - mg):", F_world[2] - weight)
            # print("slide_x:", d.qpos[slide_x_dof])
            # print("-------------------")
            # ---------------------------------------------
            
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

                    jump_started = False
                    jump_phase = "ground"

                    # if jump_count >= 3:
                    #     break

            else:
                if jump_started and jump_phase == "ground_contact":
                    jump_phase = "air"

                if jump_started and jump_phase == "air":
                    current_jump_max_height = max(
                        current_jump_max_height,
                        d.qpos[slide_z_dof]
                    )

            # -----------------------
            # Control after stabilization
            # -----------------------
            # if phase == "LOAD":
            #     controller.ori_theta = 0.0
            if on_ground:
                tau = controller.joint_torque(action)
                hip_left_torque  = np.clip(tau[0], -hip_peak_torque, hip_peak_torque)
                hip_right_torque = np.clip(tau[1], -hip_peak_torque, hip_peak_torque)

            else:
                hip_left_error  = q1_l - d.qpos[hip_left_dof]
                hip_right_error = q1_r - d.qpos[hip_right_dof]
                q_l = d.qpos[m.joint("hip_left").qposadr[0]]
                q_r = d.qpos[m.joint("hip_right").qposadr[0]]

                qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
                qd_r = d.qvel[m.joint("hip_right").dofadr[0]]
                hip_left_torque = kp*(q1_l - q_l) - kd*qd_l
                hip_right_torque = kp*(q1_r - q_r) - kd*qd_r

                # hip_left_torque  = 100*hip_left_error  - 10*d.qvel[hip_left_dof]
                # hip_right_torque = 100*hip_right_error - 10*d.qvel[hip_right_dof]

            d.ctrl[hip_left_actuator_id]  = hip_left_torque
            d.ctrl[hip_right_actuator_id] = hip_right_torque

            print("Motor torques:")
            print("  left :", hip_left_torque)
            print("  right:", hip_right_torque)

            # --- Joint reaction torques (actual) ---
            print("Joint qfrc_actuator:")
            print("  left :", d.qfrc_actuator[hip_left_dof])
            print("  right:", d.qfrc_actuator[hip_right_dof])

            # --- Ground Reaction Forces from MuJoCo ---
            total_grf = np.zeros(3)

            for i in range(d.ncon):
                con = d.contact[i]
                force = np.zeros(6)
                mj.mj_contactForce(m, d, i, force)

                # Contact force is in contact frame, convert to world
                # Contact frame normal is in con.frame
                contact_frame = con.frame.reshape(3,3)
                f_world = contact_frame @ force[:3]

                total_grf += f_world

            print("GRF world:", total_grf)
            print("GRF vertical:", total_grf[2])
            print("mg:", mj.mj_getTotalmass(m)*9.81)

            # --- Base acceleration ---
            print("Base z accel:", d.qacc[slide_z_dof])
            print("Base z vel  :", d.qvel[slide_z_dof])

            # print("=================================================")
            

            mj.mj_step(m, d)
            viewer.sync()

        # -----------------------
        # Handle unfinished jump
        # -----------------------
        if jump_started and jump_count < 3:
            jump_count += 1
            jump_x_end = d.qpos[slide_x_dof]
            height = current_jump_max_height
            euclidean_value = np.sqrt(height**2 + jump_x_end**2)
            jump_results.append((height, jump_x_end, euclidean_value))
        
        # print("\nAll jumps complete.")
        # print("Results:", jump_results)
        #return jump_results, jump_count

def compute_ground_reaction_force(m, d):

    ground_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")

    total_grf = np.zeros(3)

    for i in range(d.ncon):

        con = d.contact[i]

        if con.geom1 == ground_id or con.geom2 == ground_id:

            force = np.zeros(6)
            mj.mj_contactForce(m, d, i, force)

            R = con.frame.reshape(3,3)

            if con.geom1 == ground_id:
                f_world = R @ force[:3]
            else:
                f_world = -R @ force[:3]

            total_grf += f_world

    return total_grf
xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar_base.xml"   # <-- path to your XML

# IK settings
ik_height = -0.6
thigh_length = 0.4
calf_length = 0.4
hip_offset = 0.05

# Torque limit
hip_peak_torque = 10000

# Spring-damper-torsion gains
# [linear_kp, linear_kd, rotational_kp]
action = np.array([150.0, 1.0, 25.0])


# -------------------------------------------------------
# RUN SIMULATION
# -------------------------------------------------------

results = run(
    xml_path=xml_path,
    action=action,
    ik_value=ik_height,
    hip_peak_torque=hip_peak_torque,
    thigh_length=thigh_length,
    calf_length=calf_length,
    hip_offset=hip_offset
)

# print("\nSimulation finished.")
# print("Returned:", results)
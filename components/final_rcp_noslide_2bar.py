import numpy as np
import mujoco as mj
import time
import sys, os
import ik_5bar as ik
import vmc_2bar_planar as vmc_rp
# import ik as ik
import random
import mujoco.viewer
# import vmc_roc_angle as vmc_ra_angle
import pandas as pd
from scipy.interpolate import interp1d

# Set fixed random seed for deterministic behavior
np.random.seed(0)
random.seed(0)

def run(xml_path, action, ik_value, hip_peak_torque, knee_peak_torque, thigh_length, calf_length, push_duration,efficiency_hip, efficiency_knee):

    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)

    m.opt.iterations = 500
    m.opt.tolerance = 1e-10

    # -----------------------
    # Joint IDs
    # -----------------------
    #based on 2bar_planar.xml
    hip_id  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hinge1")
    knee_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hinge2")

    hip_dof  = m.jnt_dofadr[hip_id]
    knee_dof = m.jnt_dofadr[knee_id]

    slide_x_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_x")
    slide_z_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_z")

    slide_x_dof = m.jnt_dofadr[slide_x_id]
    slide_z_dof = m.jnt_dofadr[slide_z_id]

    # -----------------------
    # Actuators
    # -----------------------
    hip_actuator_id  = m.actuator('torque1').id
    knee_actuator_id = m.actuator('torque2').id

    # -----------------------
    # Spawn from IK
    # -----------------------
    q1, q2 = ik.ik_2r(0.0, ik_value, thigh_length, calf_length, -1)
    d.qpos[m.joint("hinge1").qposadr[0]]  = q1
    d.qpos[m.joint("hinge2").qposadr[0]]  = q2

    # Move root so foot touches ground
    d.qpos[m.joint("slide_z").qposadr[0]] = -ik_value
    #mujoco.mj_forward(m, d)

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
    current_jump_energy = 0
    current_jump_mech_energy = 0
    current_jump_joule = 0
    hip_energy = 0
    current_jump_start_time = 0
    current_jump_avg_vel = 0
    avg_vz = 0
    current_jump_start_z = 0
    time_at_max_height = 0


    start = time.time()
    kp = 300
    kd = 30
    controller = vmc_rp.Controller(xml_path, m, d, action)
    phase = "LOAD"      # LOAD → PUSH → FLIGHT → RESET
    push_time = 0.0
    push_duration = push_duration   # 80 ms push window
    lean_angle = 0.0   
    max_sim_time =8.0    # forward lean only during push
    t_elapsed = 0.0
    hip_torque = 0.0
    knee_torque = 0.0
    while t_elapsed < max_sim_time and jump_count < 3:
            

            # if time.time() - start > max_sim_time:
            #     break
            step_start = time.time()

            t_elapsed = time.time() - start
            #sim_time = d.time

            # ===================================================
            # 5 SECOND IK STABILIZATION (ADDED BLOCK)
            # ===================================================
            if t_elapsed < 2.0:

                q_h = d.qpos[m.joint("hinge1").qposadr[0]]
                q_k = d.qpos[m.joint("hinge2").qposadr[0]]

                qd_h = d.qvel[m.joint("hinge1").dofadr[0]]
                qd_k = d.qvel[m.joint("hinge2").dofadr[0]]

                d.ctrl[0] = kp*(q1 - q_h) - kd*qd_h
                d.ctrl[1] = kp*(q2 - q_k) - kd*qd_k             
                

                mj.mj_step(m, d)
                
                
                continue
            # ===================================================

            
            contact_force = controller.get_ground_contact_forces()

            forces = np.array([0,0,0])
            if len(contact_force) > 0:
                forces = contact_force[0][0]
            
                        # ---------------- DEBUG BLOCK ----------------
            alpha = controller.equivalent_orientation()
            #F_world = controller.force_world(action)
            #Fl = controller.total_linear_force(action)

            mass = mj.mj_getTotalmass(m)
            weight = mass * 9.81          

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
                    current_jump_start_time = t_elapsed
                    current_jump_start_z = d.qpos[slide_z_dof]
                    current_jump_energy = 0
                    current_jump_mech_energy = 0
                    current_jump_joule = 0

                elif jump_started and jump_phase == "air" and base_vel_z < 0:
                    jump_phase = "landing"

                elif jump_started and jump_phase == "landing" and base_vel_z >= 0:
                    jump_count += 1

                    jump_x_end = d.qpos[slide_x_dof]
                    jump_z_end = d.qpos[slide_z_dof]
                    jump_end_time = t_elapsed
                    jump_distance = jump_x_end - current_jump_start_x
                    jump_duration = jump_end_time - current_jump_start_time
                    jump_dz = jump_z_end - current_jump_start_z

                    if jump_duration > 0:
                        current_jump_avg_vel = jump_distance / jump_duration
                        avg_vz = jump_dz / jump_duration
                    else:
                        current_jump_avg_vel = 0
                        avg_vz = 0
                    height = current_jump_max_height

                    euclidean_value = np.sqrt(height**2 + jump_x_end**2)

                    jump_results.append((height, jump_x_end, euclidean_value, current_jump_energy, current_jump_mech_energy, current_jump_joule, jump_distance, jump_duration, current_jump_avg_vel, avg_vz))

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

            base_x_vel = d.qvel[slide_x_dof]
            base_z_vel = d.qvel[slide_z_dof]

# ---------------- PHASE MACHINE ----------------

            if phase == "LOAD":

                # Vertical leg, no lean
                controller.ori_theta = 0.0

                if on_ground:
                    tau = controller.joint_torque(action)
                    hip_torque  = np.clip(tau[0], -hip_peak_torque, hip_peak_torque)
                    knee_torque = np.clip(tau[1], -knee_peak_torque, knee_peak_torque)

                    # detect compression (velocity near zero upward)
                    if base_z_vel >= 0:
                        phase = "PUSH"
                        push_time = 0.0
                else:
                    phase = "FLIGHT"

            elif phase == "PUSH":

                controller.ori_theta = lean_angle

                tau = controller.joint_torque(action)
                hip_torque  = np.clip(tau[0], -hip_peak_torque, hip_peak_torque)
                knee_torque = np.clip(tau[1], -knee_peak_torque, knee_peak_torque)

                push_time += m.opt.timestep

                if push_time > push_duration:
                    phase = "FLIGHT"

            elif phase == "FLIGHT":

                controller.ori_theta = 0.0

                # retract to neutral pose
                q_h = d.qpos[m.joint("hinge1").qposadr[0]]
                q_k = d.qpos[m.joint("hinge2").qposadr[0]]

                qd_h = d.qvel[m.joint("hinge1").dofadr[0]]
                qd_k = d.qvel[m.joint("hinge2").dofadr[0]]

                hip_torque  = np.clip(kp*(q1 - q_h) - kd*qd_h, -hip_peak_torque, hip_peak_torque)
                knee_torque = np.clip(kp*(q2 - q_k) - kd*qd_k, -knee_peak_torque, knee_peak_torque)

                if on_ground:
                    phase = "RESET"

            elif phase == "RESET":

                controller.ori_theta = 0.0

                # hold vertical for stabilization
                q_h = d.qpos[m.joint("hinge1").qposadr[0]]
                q_k = d.qpos[m.joint("hinge2").qposadr[0]]

                qd_h = d.qvel[m.joint("hinge1").dofadr[0]]
                qd_k = d.qvel[m.joint("hinge2").dofadr[0]]

                hip_torque  = np.clip(kp*(q1 - q_h) - kd*qd_h, -hip_peak_torque, hip_peak_torque)
                knee_torque = np.clip(kp*(q2 - q_k) - kd*qd_k, -knee_peak_torque, knee_peak_torque)

                # once stabilized and compressing again → next jump
                if base_z_vel >= 0:
                    phase = "LOAD"

            # ------------------------------------------------

            d.ctrl[hip_actuator_id]  = efficiency_hip*hip_torque
            d.ctrl[knee_actuator_id] = efficiency_knee*knee_torque

            if jump_started and t_elapsed > 2.0 and phase in ["PUSH"]:
                # Mechanical energy (work done by actuators)
                ha = m.opt.timestep * hip_torque * d.qvel[hip_dof]*efficiency_hip            
                ka = m.opt.timestep * knee_torque * d.qvel[knee_dof]*efficiency_knee
                if ha < 0:
                    ha = 0
                if ka < 0:
                    ka = 0
                mech_energy_step = ha + ka
                kv = 100  # RPM per Volt``
                kv_new = (kv*2*np.pi)/60
                kt = 1/kv_new
                R = 0.186
                # Joule heating energy (I²R losses)
                hip1_joule = (kv_new*kv_new)* (efficiency_hip**2)*(hip_torque**2) * R * m.opt.timestep
                hip2_joule = (kv_new*kv_new)* (efficiency_knee**2)*(knee_torque**2) * R * m.opt.timestep

                joule_heating_power = (kv_new*kv_new)*(efficiency_hip**2 + efficiency_knee**2)*(hip_torque**2 + knee_torque**2)*R
                #print(f"hip1_torque: {hip_torque:.2f}, hip2_torque: {knee_torque:.2f}, mech_energy_step: {mech_energy_step:.4f} J, hip1_joule: {hip1_joule:.4f} J, hip2_joule: {hip2_joule:.4f} J ")
                # if jump_count > 1:
                #     print(f"Joule heating power: {joule_heating_power:.2f} W")
                joule_energy_step = hip1_joule + hip2_joule
                
                # Total energy (mechanical + joule)
                total_energy_step = mech_energy_step
                
                # Update jump energy accumulators
                current_jump_mech_energy += mech_energy_step
                current_jump_joule += joule_energy_step
                current_jump_energy += total_energy_step
                
                # Also track total energy for all jumps
                hip_energy += ha + ka
                #print(f"slide x vel: {base_x_vel:.4f} m/s")
                
            

            mj.mj_step(m, d)
            # time_until_next_step = m.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)

        # -----------------------
        # Handle unfinished jump
        # -----------------------
    if len(jump_results) == 0:

        best_height = -1e6
        best_x_vel = 0
        best_distance = 0
        best_energy = 1e6
        best_duration = 0

    else:

        # choose jump with highest forward velocity
        best_jump = max(jump_results, key=lambda x: x[8])

        best_height = best_jump[0]
        best_distance = best_jump[6]
        best_energy = best_jump[3]
        best_duration = best_jump[7]
        best_x_vel = best_jump[8]

    return (
        best_height,
        best_x_vel,
        best_distance,
        best_energy,
        best_duration,
        jump_results
    )
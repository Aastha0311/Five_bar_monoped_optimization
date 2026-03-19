import numpy as np
import mujoco as mj
import time
import sys, os
import ik_5bar as ik
import vmc_action_5bar as vmc_rp
# import ik as ik
import random
import mujoco.viewer
# import vmc_roc_angle as vmc_ra_angle
import pandas as pd
from scipy.interpolate import interp1d
import imageio

# Set fixed random seed for deterministic behavior
np.random.seed(0)
random.seed(0)

def run(xml_path, action, ik_value, hip1_peak_torque, hip2_peak_torque, thigh_length, calf_length, hip_offset, push_duration,efficiency_left, efficiency_right):

    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)

    m.opt.iterations = 500
    m.opt.tolerance = 1e-10
    # record_video = True
    # video_filename = "5bar_test_jump.mp4"
    # video_fps = 30

    # if record_video:
    #     renderer = mj.Renderer(m, width=640, height=480)
    #     frames = []

    # -----------------------
    # Joint IDs
    # -----------------------
    hip_left_id  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_left")
    hip_right_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_right")

    hip_left_dof  = m.jnt_dofadr[hip_left_id] #start address in qvel array for this joint
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
    #d.qpos[m.joint("slide_z").qposadr[0]] = -ik_value    
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
    controller = vmc_rp.Controller(m, d, xml_path, action)
    phase = "LOAD"      # LOAD → PUSH → FLIGHT → RESET
    push_time = 0.0
    push_duration = push_duration   # 80 ms push window
    lean_angle = 0.0  
    max_sim_time = 30.0    # forward lean only during push
    t_elapsed = 0.0

    # with mujoco.viewer.launch_passive(m, d) as viewer:
    #     while viewer.is_running():
            #while t_elapsed < max_sim_time and jump_count < 3:
    while jump_count < 1:

            # if time.time() - start > max_sim_time:
            #     break
    

            t_elapsed = time.time() - start
            #sim_time = d.time

            # ===================================================
            # 5 SECOND IK STABILIZATION (ADDED BLOCK)
            # ===================================================
            if t_elapsed < 0.05:

                q_l = d.qpos[m.joint("hip_left").qposadr[0]]
                q_r = d.qpos[m.joint("hip_right").qposadr[0]]

                qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
                qd_r = d.qvel[m.joint("hip_right").dofadr[0]]

                d.ctrl[0] = kp*(q1_l - q_l) - kd*qd_l
                d.ctrl[1] = kp*(q1_r - q_r) - kd*qd_r             
            
                
                

                mj.mj_step(m, d)
                #viewer.sync()
                
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
                        current_jump_avg_vel = abs(current_jump_avg_vel)
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
                    tau = controller.joint_torque()
                    hip_left_torque  = np.clip(tau[0], -hip1_peak_torque, hip1_peak_torque)
                    hip_right_torque = np.clip(tau[1], -hip2_peak_torque, hip2_peak_torque)

                    # detect compression (velocity near zero upward)
                    if base_z_vel >= 0:
                        phase = "PUSH"
                        push_time = 0.0
                else:
                    phase = "FLIGHT"

            elif phase == "PUSH":

                controller.ori_theta = lean_angle

                tau = controller.joint_torque()
                hip_left_torque  = np.clip(tau[0], -hip1_peak_torque, hip1_peak_torque)
                hip_right_torque = np.clip(tau[1], -hip2_peak_torque, hip2_peak_torque)

                push_time += m.opt.timestep

                if push_time > push_duration:
                    phase = "FLIGHT"

            elif phase == "FLIGHT":

                controller.ori_theta = 0.0

                # retract to neutral pose
                q_l = d.qpos[m.joint("hip_left").qposadr[0]]
                q_r = d.qpos[m.joint("hip_right").qposadr[0]]

                qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
                qd_r = d.qvel[m.joint("hip_right").dofadr[0]]

                hip_left_torque  = np.clip(kp*(q1_l - q_l) - kd*qd_l, -hip1_peak_torque, hip1_peak_torque)
                hip_right_torque = np.clip(kp*(q1_r - q_r) - kd*qd_r, -hip2_peak_torque, hip2_peak_torque)

                if on_ground:
                    phase = "RESET"

            elif phase == "RESET":

                controller.ori_theta = 0.0

                # hold vertical for stabilization
                q_l = d.qpos[m.joint("hip_left").qposadr[0]]
                q_r = d.qpos[m.joint("hip_right").qposadr[0]]

                qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
                qd_r = d.qvel[m.joint("hip_right").dofadr[0]]

                hip_left_torque  = np.clip(kp*(q1_l - q_l) - kd*qd_l, -hip1_peak_torque, hip1_peak_torque)
                hip_right_torque = np.clip(kp*(q1_r - q_r) - kd*qd_r, -hip2_peak_torque, hip2_peak_torque)

                # once stabilized and compressing again → next jump
                if base_z_vel >= 0:
                    phase = "LOAD"

            # ------------------------------------------------

            d.ctrl[hip_left_actuator_id]  = efficiency_left*hip_left_torque
            d.ctrl[hip_right_actuator_id] = efficiency_right*hip_right_torque

            if jump_started and phase in ["PUSH"]:
                # Mechanical energy (work done by actuators)
                ha1 = m.opt.timestep * hip_left_torque * d.qvel[hip_left_dof]            
                ha2 = m.opt.timestep * hip_right_torque * d.qvel[hip_right_dof]
                if ha1 < 0:
                    ha1 = 0
                if ha2 < 0:
                    ha2 = 0
                mech_energy_step = ha1 + ha2
                kv = 100  # RPM per Volt``
                kv_new = (kv*2*np.pi)/60
                kt = 1/kv_new
                R = 0.186
                # Joule heating energy (I²R losses)
                hip1_joule = (kv_new*kv_new)* (efficiency_left**2)*(hip_left_torque**2) * R * m.opt.timestep
                hip2_joule = (kv_new*kv_new)* (efficiency_right**2)*(hip_right_torque**2) * R * m.opt.timestep

                joule_heating_power = (kv_new*kv_new)*(efficiency_left**2 + efficiency_right**2)*(hip_left_torque**2 + hip_right_torque**2)*R
                #print(f"hip1_torque: {hip_left_torque:.2f}, hip2_torque: {hip_right_torque:.2f}, mech_energy_step: {mech_energy_step:.4f} J, hip1_joule: {hip1_joule:.4f} J, hip2_joule: {hip2_joule:.4f} J ")
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
                hip_energy += ha1 + ha2
                #print(f"slide x vel: {base_x_vel:.4f} m/s")
                
            

            mj.mj_step(m, d)
            #viewer.sync()
            # if record_video:
            #     renderer.update_scene(d)
            #     frame = renderer.render()
            #     frames.append(frame.copy())
            

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

    # if record_video:
    #     renderer.close()
    #     imageio.mimsave(video_filename, frames, fps=video_fps)
    #     print(f"Video saved to {video_filename}")


    return (
        best_height,
        best_x_vel,
        best_distance,
        best_energy,
        best_duration,
        jump_results
    )
                

    # return jump_results
        
        


xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/design_xmls/0dd5167c.xml"   # <-- path to your XML

# # IK settings
ik_height = -0.3
thigh_length = 0.22252482719749359
calf_length = 0.206840194636663
hip_offset = 0.050793396782675845
efficiency_left = 0.9
efficiency_right = 0.9
# Torque limit
hip_peak_torque = 10000

# Spring-damper-torsion gains
# [linear_kp, linear_kd, rotational_kp]
action = np.array([100.0, 1.0, 25.0])


# -------------------------------------------------------
# RUN SIMULATION
# -------------------------------------------------------

# results = run(
#     xml_path=xml_path,
#     action=action,
#     ik_value=ik_height,
#     hip1_peak_torque=hip_peak_torque,
#     hip2_peak_torque=hip_peak_torque,
#     thigh_length=thigh_length,
#     calf_length=calf_length,
#     hip_offset=hip_offset,
#     push_duration=0.1,
#     kv=100,
#     R= 0.186  # 80 ms push
# )

results = run(xml_path, action, ik_value=ik_height, hip1_peak_torque=hip_peak_torque,
    hip2_peak_torque=hip_peak_torque, thigh_length=thigh_length,
    calf_length=calf_length,
    hip_offset=hip_offset,
    push_duration=0.1, efficiency_left=efficiency_left, efficiency_right=efficiency_right)

print("Best Jump Height: {:.4f} m".format(results[0]))
print("Best Jump Forward Velocity: {:.4f} m/s".format(results[1]))
print("Best Jump Distance: {:.4f} m".format(results[2]))
print("Best Jump Energy: {:.4f} J".format(results[3]))
print("Best Jump Duration: {:.4f} s".format(results[4]))


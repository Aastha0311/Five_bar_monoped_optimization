import numpy as np
import mujoco as mj
import time
import sys, os
#sys.path.append("/home/stochlab/repo/Aastha_Coopt_Monoped/Monoped-optimization")
import ik_5bar as ik
import vmc_action_5bar as vmc_rp
import imageio
# import ik as ik
import random
# import vmc_roc_angle as vmc_ra_angle
import pandas as pd
from scipy.interpolate import interp1d
# Set fixed random seed for deterministic behavior
np.random.seed(0)
random.seed(0)
import mujoco.viewer

def run(xml_path, action, ik_value, hip1_peak_torque, hip2_peak_torque, thigh_length, calf_length, hip_offset, efficiency_left, efficiency_right, kt_left, kt_right, r_left, r_right):
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)

    m.opt.iterations = 500  
    m.opt.tolerance = 1e-10

    # record_video = True
    # video_filename = "5bar_old_planar_jump.mp4"
    # video_fps = 10
    # if record_video:
    #     renderer = mj.Renderer(m, width=640, height=480)
    #     frames = []

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
    total_energy = 0
    hip_energy = 0
    knee_energy = 0
    jump_energy = 0  
    jump_results = []  
    current_jump_max = float('-inf')
    current_jump_x_end = 0
    max_height_after_control = float('-inf')
    jump_count = 0
    in_air = False
    tracking_jump = False
    # hip_actuator_id = m.actuator('torque1').id    
    # knee_actuator_id = m.actuator('torque2').id    
    l1n = thigh_length
    l2n = calf_length
    r_left = r_left
    r_right = r_right

    # theta1, theta2 = ik.inverse_kinematics(0, ik_value, l1n, l2n)
    
    # theta1d = np.rad2deg(theta1)
    # theta2d = np.rad2deg(theta2)
    current_x_max = float('-inf')
    start = time.time() 
 
    # Jump tracking variables
    jump_started = False
    jump_phase = "ground"  # "ground", "takeoff", "air", "landing"
    current_jump_mech_energy = 0
    current_jump_joule_energy = 0
    current_jump_total_energy = 0
    current_jump_max_z = float('-inf')
    previous_base_vel = 0
    current_jump_start_x = 0
    current_jump_start_time = 0
    current_jump_avg_vel = 0
    kp = 200
    kd = 20
    
    # Joule heating constant
    kt = 0.0955

    max_steps = 2000
    step_counter = 0
    # with mujoco.viewer.launch_passive(m, d) as viewer:
    #     while viewer.is_running():
    while jump_count < 1 and step_counter < max_steps:
        step_counter += 1  # Stop after 1 jump
        step_start = time.time()
        t_elapsed = time.time() - start        
        if t_elapsed < 0:
            # hip_error = 0 - d.qpos[2]
            # knee_error = 0 - d.qpos[3]
            # hip_torque = 100 * hip_error - 10 * d.qvel[2]
            # knee_torque = 100 * knee_error - 10 * d.qvel[3]
            q_l = d.qpos[m.joint("hip_left").qposadr[0]]
            q_r = d.qpos[m.joint("hip_right").qposadr[0]]

            qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
            qd_r = d.qvel[m.joint("hip_right").dofadr[0]]

            hip_left_torque = kp*(q1_l - q_l) - kd*qd_l
            hip_right_torque = kp*(q1_r - q_r) - kd*qd_r

            #mj.mj_step(m, d)
            #viewer.sync()
        else:
            controller = vmc_rp.Controller(m, d, xml_path, action)
            contact_force = controller.get_ground_contact_forces()
            forces = np.array([0, 0, 0])  
            if len(contact_force) > 0:
                forces = contact_force[0][0]
            
            # Get current base velocity
            current_base_vel = d.qvel[slide_z_dof]
            on_ground = not np.all(forces == 0)
            
            # Jump phase detection logic
            if on_ground:
                if not jump_started and current_base_vel > 0:
                    # Start of jump: positive velocity while in contact with ground
                    jump_started = True
                    jump_phase = "ground_contact"
                    current_jump_mech_energy = 0
                    current_jump_joule_energy = 0
                    current_jump_total_energy = 0
                    current_jump_max_z = d.qpos[slide_z_dof]
                    current_jump_start_x = d.qpos[slide_x_dof]
                    current_jump_start_z = d.qpos[slide_z_dof]
                    current_jump_start_time = t_elapsed
                
                elif jump_started and jump_phase == "air" and current_base_vel < 0:
                    # Landing phase: negative velocity while landing
                    jump_phase = "landing"
                
                elif jump_started and jump_phase == "landing" and current_base_vel >= 0:
                    # Jump completed, record results
                    jump_count += 1
                    current_jump_x_end = d.qpos[slide_x_dof]
                    jump_z_end = d.qpos[slide_z_dof]
                    jump_distance = current_jump_x_end - current_jump_start_x
                    jump_duration = t_elapsed - current_jump_start_time
                    jump_dz = jump_z_end - current_jump_start_z
                    if jump_duration > 0:
                        current_jump_avg_vel = jump_distance / jump_duration
                        current_jump_avg_vel = abs(current_jump_avg_vel)
                        avg_vz = jump_dz / jump_duration
                    else:
                        current_jump_avg_vel = 0
                        avg_vz = 0
                    euclidean_distance = np.sqrt(jump_distance**2 + current_jump_max_z**2)
                    height = current_jump_max_z
                    jump_results.append((height, current_jump_x_end, euclidean_distance, current_jump_total_energy, current_jump_mech_energy, current_jump_joule_energy, jump_distance, jump_duration, current_jump_avg_vel, avg_vz))

                    max_height_after_control = max(max_height_after_control, current_jump_max_z)
                    
                    # Reset for next jump
                    jump_started = False
                    jump_phase = "ground"
                    current_jump_mech_energy = 0
                    current_jump_joule_energy = 0
                    current_jump_total_energy = 0
                    current_jump_max_z = float('-inf')
                    
                    # Break if we've reached 3 jumps
                    # if jump_count >= 3:
                    #     break
                
                # Apply controller forces when on ground and jump started
                if jump_started:                    
                    joint_torque = controller.joint_torque()
                    hip_left_torque = np.clip(joint_torque[0], -hip1_peak_torque, hip1_peak_torque)
                    hip_right_torque = np.clip(joint_torque[1], -hip2_peak_torque, hip2_peak_torque)
                else:
                    # Default control when not in a jump phase
                    # hip_error = 0 - d.qpos[2]
                    # knee_error = 0 - d.qpos[3]
                    # hip_torque = 100 * hip_error - 10 * d.qvel[2]
                    # knee_torque = 100 * knee_error - 10 * d.qvel[3]
                        q_l = d.qpos[m.joint("hip_left").qposadr[0]]
                        q_r = d.qpos[m.joint("hip_right").qposadr[0]]

                        qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
                        qd_r = d.qvel[m.joint("hip_right").dofadr[0]]

                        hip_left_torque  = np.clip(kp*(q1_l - q_l) - kd*qd_l, -hip1_peak_torque, hip1_peak_torque)
                        hip_right_torque = np.clip(kp*(q1_r - q_r) - kd*qd_r, -hip2_peak_torque, hip2_peak_torque)
            
            else:  # In air
                if jump_started and jump_phase == "ground_contact":
                    # Transition from ground to air (takeoff)
                    jump_phase = "air"
                
                elif jump_started and jump_phase == "air":
                    # Update max height during air phase
                    current_jump_max_z = max(current_jump_max_z, d.qpos[slide_z_dof])
                
                # Default control in air
                q_l = d.qpos[m.joint("hip_left").qposadr[0]]
                q_r = d.qpos[m.joint("hip_right").qposadr[0]]

                qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
                qd_r = d.qvel[m.joint("hip_right").dofadr[0]]

                hip_left_torque  = np.clip(kp*(q1_l - q_l) - kd*qd_l, -hip1_peak_torque, hip1_peak_torque)
                hip_right_torque = np.clip(kp*(q1_r - q_r) - kd*qd_r, -hip2_peak_torque, hip2_peak_torque)
            
            previous_base_vel = current_base_vel
        
        # Apply Control Torques
        hip_left_torque = np.clip(hip_left_torque, -hip1_peak_torque, hip1_peak_torque)
        hip_right_torque = np.clip(hip_right_torque, -hip2_peak_torque, hip2_peak_torque)
        
        d.ctrl[hip_left_actuator_id] = efficiency_left * hip_left_torque
        d.ctrl[hip_right_actuator_id] = efficiency_right * hip_right_torque

        
        # Calculate energy for the current jump (starts from ground contact with positive velocity)
        if jump_started and t_elapsed >= 0:
            # Mechanical energy (work done by actuators)
            ha1 = m.opt.timestep * efficiency_left*hip_left_torque * d.qvel[hip_left_dof]            
            ha2 = m.opt.timestep * efficiency_right*hip_right_torque * d.qvel[hip_right_dof]
            if ha1 < 0:
                ha1 = 0
            if ha2 < 0:
                ha2 = 0
            mech_energy_step = ha1 + ha2
            
            # Joule heating energy (I²R losses)
            hip_left_joule = (1/kt_left) * kt_left* (efficiency_left*hip_left_torque**2) * m.opt.timestep*r_left
            hip_right_joule = (1/kt_right) * kt_right* (efficiency_right*hip_right_torque**2) * m.opt.timestep*r_right
            joule_energy_step = hip_left_joule + hip_right_joule
            
            # Total energy (mechanical + joule)
            total_energy_step = mech_energy_step + joule_energy_step
            
            # Update jump energy accumulators
            current_jump_mech_energy += mech_energy_step
            current_jump_joule_energy += joule_energy_step
            current_jump_total_energy += total_energy_step
            
            # Also track total energy for all jumps
            hip_energy += ha1
            knee_energy += ha2
            total_energy += mech_energy_step
        
        # Step simulation
    
        
        mj.mj_step(m, d)
        # viewer.sync()
        # if record_video:
        #         renderer.update_scene(d)
        #         frame = renderer.render()
        #         frames.append(frame.copy())
        
        
        # Respect simulation timestep
        # time_until_next_step = m.opt.timestep - (time.time() - step_start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)

    # Record the final jump if simulation ends during a jump
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
xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/design_xmls/8778abeb.xml"   # <-- path to your XML

# # # IK settings
ik_height = -0.4203276682254768
thigh_length = 0.3046289869522943
calf_length = 0.190030742101534
hip_offset = 0.0500076970975257*0.5
efficiency_left = 0.963
efficiency_right = 0.939
# Torque limit
hip2_peak_torque = 9.04999755345784*2.304
hip1_peak_torque = 2.292*4.000092005818691

# Spring-damper-torsion gains
# [linear_kp, linear_kd, rotational_kp]
action = np.array([357.6, 9.9,29.1])


results = run(xml_path, action, ik_value=ik_height, hip1_peak_torque=hip1_peak_torque,
    hip2_peak_torque=hip2_peak_torque, thigh_length=thigh_length,
    calf_length=calf_length,
    hip_offset=hip_offset, efficiency_left=efficiency_left, efficiency_right=efficiency_right)

print("Best Jump Height: {:.4f} m".format(results[0]))
print("Best Jump Forward Velocity: {:.4f} m/s".format(results[1]))
print("Best Jump Distance: {:.4f} m".format(results[2]))
print("Best Jump Energy: {:.4f} J".format(results[3]))
print("Best Jump Duration: {:.4f} s".format(results[4]))
import numpy as np
import mujoco as mj
import time
import sys, os
sys.path.append("/home/stochlab/repo/Aastha_Coopt_Monoped/Monoped-optimization")
import ik_5bar as ik
import vmc_2bar_planar as vmc_rp

# import ik as ik
import random
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
    
    
    total_energy = 0
    hip_energy = 0
    knee_energy = 0
    jump_energy = 0  
    jump_results = []  
    #current_jump_max = float('-inf')
    current_jump_x_end = 0
    max_height_after_control = float('-inf')
    jump_count = 0
    in_air = False
    tracking_jump = False
    hip_actuator_id = m.actuator('torque1').id    
    knee_actuator_id = m.actuator('torque2').id   
    
    
    
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
    
    # Joule heating constant
    kt = 0.0955
    push_time = 0.0
    push_duration = push_duration   # 80 ms push window
    lean_angle = 0.0   
    max_sim_time = 8.0    # forward lean only during push
    t_elapsed = 0.0
    hip_torque = 0.0
    knee_torque = 0.0
    kp = 300
    kd = 30
    
    while t_elapsed < max_sim_time and jump_count < 3:  # Stop after 3 jumps
        step_start = time.time()
        t_elapsed = time.time() - start        
        if t_elapsed < 2:
            q_h = d.qpos[m.joint("hinge1").qposadr[0]]
            q_k = d.qpos[m.joint("hinge2").qposadr[0]]

            qd_h = d.qvel[m.joint("hinge1").dofadr[0]]
            qd_k = d.qvel[m.joint("hinge2").dofadr[0]]

            d.ctrl[0] = kp*(q1 - q_h) - kd*qd_h
            d.ctrl[1] = kp*(q2 - q_k) - kd*qd_k             
            

            mj.mj_step(m, d)
            
            
            continue
        else:
            controller = vmc_rp.Controller(xml_path, m, d, action)
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
                        avg_vz = jump_dz / jump_duration
                    else:
                        current_jump_avg_vel = 0
                        avg_vz = 0
                    
                    
                    euclidean_distance = np.sqrt(jump_distance**2 + current_jump_max_z**2)
                    jump_results.append((current_jump_max_z, current_jump_x_end, euclidean_distance, current_jump_total_energy, current_jump_mech_energy, current_jump_joule_energy, jump_distance, jump_duration, current_jump_avg_vel, avg_vz))
                    max_height_after_control = max(max_height_after_control, current_jump_max_z)
                    
                    # Reset for next jump
                    jump_started = False
                    jump_phase = "ground"
                    # current_jump_mech_energy = 0
                    # current_jump_joule_energy = 0
                    # current_jump_total_energy = 0
                    # current_jump_max_z = float('-inf')
                    
                    # Break if we've reached 3 jumps
                    # if jump_count >= 3:
                    #     break
                
                # Apply controller forces when on ground and jump started
                if jump_started:
                    
                    joint_torque = controller.joint_torque(action)
                    hip_torque = np.clip(joint_torque[0], -hip_peak_torque, hip_peak_torque)
                    knee_torque = np.clip(joint_torque[1], -knee_peak_torque, knee_peak_torque)
                else:
                    # Default control when not in a jump phase
                    q_h = d.qpos[m.joint("hinge1").qposadr[0]]
                    q_k = d.qpos[m.joint("hinge2").qposadr[0]]

                    qd_h = d.qvel[m.joint("hinge1").dofadr[0]]
                    qd_k = d.qvel[m.joint("hinge2").dofadr[0]]
                    hip_torque = kp*(q1 - q_h) - kd*qd_h
                    knee_torque = kp*(q2 - q_k) - kd*qd_k
            
            else:  # In air
                if jump_started and jump_phase == "ground_contact":
                    # Transition from ground to air (takeoff)
                    jump_phase = "air"
                
                elif jump_started and jump_phase == "air":
                    # Update max height during air phase
                    current_jump_max_z = max(current_jump_max_z, d.qpos[slide_z_dof])
                
                # Default control in air
                q_h = d.qpos[m.joint("hinge1").qposadr[0]]
                q_k = d.qpos[m.joint("hinge2").qposadr[0]]

                qd_h = d.qvel[m.joint("hinge1").dofadr[0]]
                qd_k = d.qvel[m.joint("hinge2").dofadr[0]]
                hip_torque = kp*(q1 - q_h) - kd*qd_h
                knee_torque = kp*(q2 - q_k) - kd*qd_k
            
            previous_base_vel = current_base_vel
        
        # Apply Control Torques
        hip_torque = np.clip(hip_torque, -hip_peak_torque, hip_peak_torque)
        knee_torque = np.clip(knee_torque, -knee_peak_torque, knee_peak_torque)
        
        d.ctrl[hip_actuator_id] = efficiency_hip * hip_torque
        d.ctrl[knee_actuator_id] = efficiency_knee * knee_torque

        
        # Calculate energy for the current jump (starts from ground contact with positive velocity)
        if jump_started and t_elapsed >= 0:
            # Mechanical energy (work done by actuators)
            ha = m.opt.timestep * hip_torque * d.qvel[hip_dof] * efficiency_hip           
            ka = m.opt.timestep * knee_torque * d.qvel[knee_dof] * efficiency_knee
            if ha < 0:
                ha = 0
            if ka < 0:
                ka = 0
            mech_energy_step = ha + ka
            
            # Joule heating energy (I²R losses)
            hip_joule = kt * (hip_torque**2) * m.opt.timestep
            knee_joule = kt * (knee_torque**2) * m.opt.timestep
            joule_energy_step = hip_joule + knee_joule
            
            # Total energy (mechanical + joule)
            total_energy_step = mech_energy_step
            
            # Update jump energy accumulators
            current_jump_mech_energy += mech_energy_step
            current_jump_joule_energy += joule_energy_step
            current_jump_total_energy += total_energy_step
            
            # Also track total energy for all jumps
            hip_energy += ha
            knee_energy += ka
            total_energy += mech_energy_step
        
        # Step simulation
    
        
        mj.mj_step(m, d)

        
        
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

    return (
        best_height,
        best_x_vel,
        best_distance,
        best_energy,
        best_duration,
        jump_results
    )
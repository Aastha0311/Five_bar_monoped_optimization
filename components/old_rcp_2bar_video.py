import numpy as np
import mujoco as mj
import time
import sys, os
sys.path.append("/home/stochlab/repo/Aastha_Coopt_Monoped/Monoped-optimization")
import ik_5bar as ik
import vmc_2bar_old as vmc_rp
# import ik as ik
import random
import mujoco.viewer
# import vmc_roc_angle as vmc_ra_angle
import pandas as pd
import imageio
from scipy.interpolate import interp1d
# Set fixed random seed for deterministic behavior
np.random.seed(0)
random.seed(0)

def run(xml_path, action, ik_value, hip_peak_torque, knee_peak_torque, thigh_length, calf_length, efficiency_hip, efficiency_knee):
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)

    
    m.opt.iterations = 500  
    m.opt.tolerance = 1e-10
    record_video = True
    video_filename = "2bar_stoch3_40.mp4"
    video_fps = 30
    if record_video:
        renderer = mj.Renderer(m, width=640, height=480)
        frames = []
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
    hip_actuator_id = m.actuator('torque1').id    
    knee_actuator_id = m.actuator('torque2').id    
    l1n = thigh_length
    l2n = calf_length

    theta1, theta2 = ik.inverse_kinematics(0, ik_value, l1n, l2n)
    
    theta1d = np.rad2deg(theta1)
    theta2d = np.rad2deg(theta2)
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
    max_steps = 2000
    step_counter = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            while jump_count < 1 and step_counter < max_steps: 
                step_counter += 1 
                step_start = time.time()
                t_elapsed = time.time() - start        
                if t_elapsed < 0:
                    hip_error = 0 - d.qpos[2]
                    knee_error = 0 - d.qpos[3]
                    hip_torque = 100 * hip_error - 10 * d.qvel[2]
                    knee_torque = 100 * knee_error - 10 * d.qvel[3]
                else:
                    controller = vmc_rp.Controller(xml_path, m, d, theta1, theta2, action)
                    contact_force = controller.get_ground_contact_forces()
                    forces = np.array([0, 0, 0])  
                    if len(contact_force) > 0:
                        forces = contact_force[0][0]
                    
                    # Get current base velocity
                    current_base_vel = d.qvel[1]
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
                            current_jump_max_z = d.qpos[1]
                            current_jump_start_z = d.qpos[1]
                            current_jump_start_x = d.qpos[0]
                            current_jump_start_time = t_elapsed
                        
                        elif jump_started and jump_phase == "air" and current_base_vel < 0:
                            # Landing phase: negative velocity while landing
                            jump_phase = "landing"
                        
                        elif jump_started and jump_phase == "landing" and current_base_vel >= 0:
                            # Jump completed, record results
                            jump_count += 1
                            current_jump_x_end = d.qpos[0]
                            current_jump_z_end = d.qpos[1]
                            jump_dz = current_jump_z_end - current_jump_start_z
                            jump_distance = current_jump_x_end - current_jump_start_x
                            jump_duration = t_elapsed - current_jump_start_time
                            euclidean_distance = np.sqrt(jump_distance**2 + current_jump_max_z**2)
                            if jump_duration > 0:
                                current_jump_avg_vel = jump_distance / jump_duration
                                current_jump_avg_vel = abs(current_jump_avg_vel)
                                avg_vz = jump_dz / jump_duration
                            else:
                                current_jump_avg_vel = 0
                                avg_vz = 0
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
                            joint_torque = controller.joint_torque(action)
                            hip_torque = np.clip(joint_torque[0], -hip_peak_torque, hip_peak_torque)
                            knee_torque = np.clip(joint_torque[1], -knee_peak_torque, knee_peak_torque)
                        else:
                            # Default control when not in a jump phase
                            hip_error = 0 - d.qpos[2]
                            knee_error = 0 - d.qpos[3]
                            hip_torque = 100 * hip_error - 10 * d.qvel[2]
                            knee_torque = 100 * knee_error - 10 * d.qvel[3]
                    
                    else:  # In air
                        if jump_started and jump_phase == "ground_contact":
                            # Transition from ground to air (takeoff)
                            jump_phase = "air"
                        
                        elif jump_started and jump_phase == "air":
                            # Update max height during air phase
                            current_jump_max_z = max(current_jump_max_z, d.qpos[1])
                        
                        # Default control in air
                        hip_error = 0 - d.qpos[2]
                        knee_error = 0 - d.qpos[3]
                        hip_torque = 100 * hip_error - 10 * d.qvel[2]
                        knee_torque = 100 * knee_error - 10 * d.qvel[3]
                    
                    previous_base_vel = current_base_vel
                
                # Apply Control Torques
                hip_torque = np.clip(hip_torque, -hip_peak_torque, hip_peak_torque)
                knee_torque = np.clip(knee_torque, -knee_peak_torque, knee_peak_torque)
                
                d.ctrl[hip_actuator_id] = efficiency_hip*hip_torque
                d.ctrl[knee_actuator_id] = efficiency_knee*knee_torque

                
                # Calculate energy for the current jump (starts from ground contact with positive velocity)
                if jump_started and t_elapsed >= 0:
                    # Mechanical energy (work done by actuators)
                    ha = m.opt.timestep * efficiency_hip*hip_torque * d.qvel[2]            
                    ka = m.opt.timestep * efficiency_knee*knee_torque * d.qvel[3]
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
                viewer.sync()
                if record_video:
                    renderer.update_scene(d)
                    frame = renderer.render()
                    frames.append(frame.copy())
                
                
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

            if record_video:
                renderer.close()
                imageio.mimsave(video_filename, frames, fps=video_fps)
                print(f"Video saved to {video_filename}")


            return (
                best_height,
                best_x_vel,
                best_distance,
                best_energy,
                best_duration,
                jump_results
            )
xml_path = "/home/stochlab/repo/optimal-design-legged-robots/xmls/stoch3_40.xml"   # <-- path to your XML

# # # IK settings
ik_height = -0.40
thigh_length = 0.297
calf_length = 0.302
hip_peak_torque = 20.626480624709636
knee_peak_torque = 28.126268979854057
efficiency_hip = 0.952
efficiency_knee = 0.952
# # Torque limit
# hip_peak_torque = 10000

# # Spring-damper-torsion gains
# # [linear_kp, linear_kd, rotational_kp]
action = np.array([550.0, 5.0, 30.0])

results = run(xml_path, action, ik_value=ik_height, hip_peak_torque=hip_peak_torque,
    knee_peak_torque=knee_peak_torque, thigh_length=thigh_length,
    calf_length=calf_length, efficiency_hip=efficiency_hip, efficiency_knee=efficiency_knee)

print("Best Jump Height: {:.4f} m".format(results[0]))
print("Best Jump Forward Velocity: {:.4f} m/s".format(results[1]))
print("Best Jump Distance: {:.4f} m".format(results[2]))
print("Best Jump Energy: {:.4f} J".format(results[3]))
print("Best Jump Duration: {:.4f} s".format(results[4]))
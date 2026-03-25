import numpy as np
import mujoco as mj
import time
import utils.ik_5bar as ik
import vmc_action_5bar as vmc_rp

def run(xml_path, action, ik_value, hip1_peak_torque, hip2_peak_torque, thigh_length, calf_length, hip_offset, efficiency_left, efficiency_right, ori_l=10.0, ori_theta=0.0):
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)

    m.opt.iterations = 500  
    m.opt.tolerance = 1e-10

    record_video = False
    video_filename = "5bar_old_planar_jump.mp4"
    video_fps = 10
    if record_video:
        renderer = mj.Renderer(m, width=640, height=480)
        frames = []

    hip_left_id  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_left")
    hip_right_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_right")

    hip_left_dof  = m.jnt_dofadr[hip_left_id]
    hip_right_dof = m.jnt_dofadr[hip_right_id] 

    slide_x_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_x")
    slide_z_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_z")

    slide_x_dof = m.jnt_dofadr[slide_x_id]
    slide_z_dof = m.jnt_dofadr[slide_z_id]

    hip_left_actuator_id  = m.actuator('motor_left').id
    hip_right_actuator_id = m.actuator('motor_right').id
    
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
    r_left = 0.186
    r_right = 0.063

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
    kp = 300
    kd = 30
    prev_torque_left = 0
    prev_torque_right = 0
    # Joule heating constant
    kt = 0.0955
    kt_left = 9.55/100
    kt_right = 9.55/150
    max_steps = 2000
    step_counter = 0
    while jump_count < 1 and step_counter < max_steps:
        step_counter += 1  # Stop after 1 jump
        step_start = time.time()
        t_elapsed = time.time() - start        
        if t_elapsed < 0:
            q_l = d.qpos[m.joint("hip_left").qposadr[0]]
            q_r = d.qpos[m.joint("hip_right").qposadr[0]]

            qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
            qd_r = d.qvel[m.joint("hip_right").dofadr[0]]

            hip_left_torque = kp*(q1_l - q_l) - kd*qd_l
            hip_right_torque = kp*(q1_r - q_r) - kd*qd_r

        else:
            controller = vmc_rp.Controller(
                m,
                d,
                xml_path,
                action,
                thigh_length,
                calf_length,
                hip_offset * 2,
                ori_l=ori_l,
                ori_theta=ori_theta,
            )
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
                
                elif jump_started and jump_phase == "landing" :
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
                    current_jump_max_z = max(current_jump_max_z, d.qpos[slide_z_dof])
                
                # Default control in air
                q_l = d.qpos[m.joint("hip_left").qposadr[0]]
                q_r = d.qpos[m.joint("hip_right").qposadr[0]]

                qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
                qd_r = d.qvel[m.joint("hip_right").dofadr[0]]

                hip_left_torque  = np.clip(kp*(q1_l - q_l) - kd*qd_l, -hip1_peak_torque, hip1_peak_torque)
                hip_right_torque = np.clip(kp*(q1_r - q_r) - kd*qd_r, -hip2_peak_torque, hip2_peak_torque)
            
            previous_base_vel = current_base_vel
        
        hip_left_torque = np.clip(hip_left_torque, -hip1_peak_torque, hip1_peak_torque)
        hip_right_torque = np.clip(hip_right_torque, -hip2_peak_torque, hip2_peak_torque)
        d.ctrl[hip_left_actuator_id] = efficiency_left * hip_left_torque
        d.ctrl[hip_right_actuator_id] = efficiency_right * hip_right_torque

        if jump_started and t_elapsed >= 0 and on_ground and current_base_vel > 0 and jump_phase != "landing" and jump_started:
            ha1 = m.opt.timestep * efficiency_left*hip_left_torque * d.qvel[hip_left_dof]            
            ha2 = m.opt.timestep * efficiency_right*hip_right_torque * d.qvel[hip_right_dof]
            if ha1 < 0:
                ha1 = 0
            if ha2 < 0:
                ha2 = 0
            mech_energy_step = ha1 + ha2
            hip_left_joule = (1/kt_left) * (1/kt_left)* ((efficiency_left*hip_left_torque)**2) * m.opt.timestep*r_left
            hip_right_joule = (1/kt_right) * (1/kt_right)* ((efficiency_right*hip_right_torque)**2) * m.opt.timestep*r_right
            joule_energy_step = hip_left_joule + hip_right_joule
            
            total_energy_step = mech_energy_step
            
            current_jump_mech_energy += mech_energy_step
            current_jump_joule_energy += joule_energy_step
            current_jump_total_energy += total_energy_step
            
            hip_energy += ha1
            knee_energy += ha2
            total_energy += mech_energy_step
        
        mj.mj_step(m, d)

    if len(jump_results) == 0:

        best_height = -1e6
        best_x_vel = 0
        best_distance = 0
        best_energy = 1e6
        best_duration = 0

    else:

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
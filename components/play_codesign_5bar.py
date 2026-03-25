import csv
import json
import numpy as np
import mujoco as mj
import time
import sys, os
sys.path.append("/home/stochlab/repo/Aastha_Coopt_Monoped/Monoped-optimization")
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

CASE = "Nominal"  # Choose: A, B, C, or Nominal
RECORD_VIDEO = True
VIDEO_DIR = "/home/stochlab/repo/optimal-design-legged-robots/results/videos"

CASE_CHOICES = {
    "A": {
        "results_json": "/home/stochlab/repo/optimal-design-legged-robots/results/Opt_design_control_parameters/CaseA_ll.json",
        "xml_dir": "/home/stochlab/repo/optimal-design-legged-robots/xmls/Case_A_xmls",
    },
    "B": {
        "results_json": "/home/stochlab/repo/optimal-design-legged-robots/results/Opt_design_control_parameters/CaseB_gear_opt.json",
        "xml_dir": "/home/stochlab/repo/optimal-design-legged-robots/xmls/Case_B_xmls",
    },
    "C": {
        "results_json": "/home/stochlab/repo/optimal-design-legged-robots/results/Opt_design_control_parameters/CaseC_full_codesign_opt.json",
        "xml_dir": "/home/stochlab/repo/optimal-design-legged-robots/xmls/Case_C_xmls",
    },
    "NOMINAL": {
        "results_json": "/home/stochlab/repo/optimal-design-legged-robots/results/Opt_design_control_parameters/Nominal.json",
        "xml_dir": "/home/stochlab/repo/optimal-design-legged-robots/xmls/Nominal_xmls",
    },
}

def run(xml_path, action, ik_value, hip1_peak_torque, hip2_peak_torque, thigh_length, calf_length, hip_offset, efficiency_left, efficiency_right, ori_l=10.0, ori_theta=0.0, case_label="Case_A"):
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)

    m.opt.iterations = 500 
    m.opt.tolerance = 1e-6

    record_video = RECORD_VIDEO
    video_filename = os.path.join(VIDEO_DIR, f"{case_label}.mp4")
    video_fps = 100
    if record_video:
        renderer = mj.Renderer(m, width=1920, height=1080)
        frames = []

    hip_left_id  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_left")
    hip_right_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hip_right")

    hip_left_dof  = m.jnt_dofadr[hip_left_id] #start address in qvel array for this joint
    hip_right_dof = m.jnt_dofadr[hip_right_id] 
    hip_left_qpos = m.jnt_qposadr[hip_left_id]
    hip_right_qpos = m.jnt_qposadr[hip_right_id]

    slide_x_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_x")
    slide_z_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_z")

    slide_x_dof = m.jnt_dofadr[slide_x_id]
    slide_z_dof = m.jnt_dofadr[slide_z_id]
    slide_x_qpos = m.jnt_qposadr[slide_x_id]
    slide_z_qpos = m.jnt_qposadr[slide_z_id]

    knee_left_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "knee_left")
    knee_right_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "knee_right")
    knee_left_qpos = m.jnt_qposadr[knee_left_id]
    knee_right_qpos = m.jnt_qposadr[knee_right_id]

    # -----------------------
    # Actuators
    # -----------------------
    hip_left_actuator_id  = m.actuator('motor_left').id
    hip_right_actuator_id = m.actuator('motor_right').id

    # -----------------------
    # Spawn from IK
    # -----------------------
    q1_l, q2_l, q1_r, q2_r = ik.ik_5bar(0, ik_value, thigh_length, calf_length, hip_offset)
    # q1_l -= np.pi/2
    # q1_r -= np.pi/2
    left_tip, right_tip = ik.fk_5bar(q1_l, q2_l, q1_r, q2_r, thigh_length, calf_length, hip_offset)

    print("FK left:", left_tip)
    print("FK right:", right_tip)
    d.qpos[m.joint("hip_left").qposadr[0]] = q1_l
    d.qpos[m.joint("knee_left").qposadr[0]] = q2_l
    d.qpos[m.joint("hip_right").qposadr[0]] = q1_r
    d.qpos[m.joint("knee_right").qposadr[0]] = q2_r
    d.qvel[:] = 0
    print("Initial base height:", d.qpos[slide_z_dof])
    mj.mj_forward(m, d)
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
    # l1n = thigh_length
    # l2n = calf_length


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
    kp = 300
    kd = 30
    
    # Joule heating constant
    kt = 0.0955
    max_steps = 2000
    step_counter = 0
    jump_timeseries = []

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
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

                    # mj.mj_step(m, d)
                    # viewer.sync()
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
                        #print("Contact forces:", forces, jump_count, d.qvel[slide_z_dof], jump_phase)
                        
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
                            print("Phase: Ground Contact - Jump started")
                        
                        elif jump_started and jump_phase == "air" and current_base_vel < 0:
                            # Landing phase: negative velocity while landing
                            jump_phase = "landing"
                            print("Phase: Landing - Jump landing detected")
                        
                        #elif jump_started and jump_phase == "landing" and current_base_vel >= 0:
                        elif jump_started and jump_phase == "landing":
                            # Jump completed, record results
                            #
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
                            print("Phase: Jump Completed - Height: {:.4f} m, Distance: {:.4f} m, Duration: {:.4f} s, Avg Vel: {:.4f} m/s".format(height, jump_distance, jump_duration, current_jump_avg_vel))
                            max_height_after_control = max(max_height_after_control, current_jump_max_z)
                            
                            # Reset for next jump
                            jump_started = False
                            jump_phase = "ground"
                            current_jump_mech_energy = 0
                            current_jump_joule_energy = 0
                            current_jump_total_energy = 0
                            current_jump_max_z = float('-inf')
                            
                            # Break if we've reached 3 jumps
                            # if jump_count >= 1:
                            #     exit(0)
                        
                        # Apply controller forces when on ground and jump started
                        if jump_started:                            
                            joint_torque = controller.joint_torque()
                            print("Raw joint torques:", joint_torque)
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
                             print("Should not be here - not in jump, on ground. q_l:", q_l, "q_r:", q_r)
                             qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
                             qd_r = d.qvel[m.joint("hip_right").dofadr[0]]
                             
                             hip_left_torque  = np.clip(kp*(q1_l - q_l) - kd*qd_l, -hip1_peak_torque, hip1_peak_torque)
                             hip_right_torque = np.clip(kp*(q1_r - q_r) - kd*qd_r, -hip2_peak_torque, hip2_peak_torque)
                    
                    else:  # In air
                        if jump_started and jump_phase == "ground_contact":
                            # Transition from ground to air (takeoff)
                            jump_phase = "air"
                            #print("q1_l:", q1_l, "q_l:", q_l, "qd_l:", qd_l)
                            print("Phase: Takeoff - Transition to air phase detected")
                            #print("q1_r:", q1_r, "q_r:", q_r, "qd_r:", qd_r)
                        elif jump_started and jump_phase == "air":
                            # Update max height during air phase
                            current_jump_max_z = max(current_jump_max_z, d.qpos[1])
                            print("Phase: Air - In air. Current max height: {:.4f} m".format(current_jump_max_z))   
                        # Default control in air
                        q_l = d.qpos[m.joint("hip_left").qposadr[0]]
                        q_r = d.qpos[m.joint("hip_right").qposadr[0]]

                        qd_l = d.qvel[m.joint("hip_left").dofadr[0]]
                        qd_r = d.qvel[m.joint("hip_right").dofadr[0]]
                        Kp_air = 50
                        Kd_air = 5
                        hip_left_torque  = np.clip(Kp_air*(q1_l - q_l) - Kd_air*qd_l, -hip1_peak_torque, hip1_peak_torque)
                        hip_right_torque = np.clip(Kp_air*(q1_r - q_r) - Kd_air*qd_r, -hip2_peak_torque, hip2_peak_torque)
                        print("Phase: In Air - Applying default PD control. q_l:", q_l, "q_r:", q_r, "qd_l:", qd_l, "qd_r:", qd_r)
                    previous_base_vel = current_base_vel
                
                # Apply Control Torques
                hip_left_torque = np.clip(hip_left_torque, -hip1_peak_torque, hip1_peak_torque)
                hip_right_torque = np.clip(hip_right_torque, -hip2_peak_torque, hip2_peak_torque)
                #print(hip_left_torque, hip_right_torque)
                d.ctrl[hip_left_actuator_id] = efficiency_left * hip_left_torque
                d.ctrl[hip_right_actuator_id] = efficiency_right * hip_right_torque
                #print(efficiency_left * hip_left_torque, efficiency_right * hip_right_torque)
                #print("efficiency left:", efficiency_left, "efficiency right:", efficiency_right)
                #print("Hip_left_torque:", hip_left_torque, "Hip_right_torque:", hip_right_torque)
                #print(d.qpos[slide_z_dof])

                if jump_started:
                    jump_timeseries.append({
                        "time": t_elapsed,
                        "slide_x": d.qpos[slide_x_qpos],
                        "slide_z": d.qpos[slide_z_qpos],
                        "left_hip": d.qpos[hip_left_qpos],
                        "left_knee": d.qpos[knee_left_qpos],
                        "right_hip": d.qpos[hip_right_qpos],
                        "right_knee": d.qpos[knee_right_qpos],
                        "ctrl_left": d.ctrl[hip_left_actuator_id],
                        "ctrl_right": d.ctrl[hip_right_actuator_id],
                    })

                
                # Calculate energy for the current jump (starts from ground contact with positive velocity)
                if jump_started and t_elapsed >= 0 and on_ground and current_base_vel > 0 and jump_phase != "landing" and jump_started:
                    # Mechanical energy (work done by actuators)
                    ha1 = m.opt.timestep * efficiency_left*hip_left_torque * d.qvel[hip_left_dof]            
                    ha2 = m.opt.timestep * efficiency_right*hip_right_torque * d.qvel[hip_right_dof]
                    if ha1 < 0:
                        ha1 = 0
                    if ha2 < 0:
                        ha2 = 0
                    mech_energy_step = ha1 + ha2
                    
                    # Joule heating energy (I²R losses)
                    hip_joule = kt * (hip_left_torque**2) * m.opt.timestep
                    knee_joule = kt * (hip_right_torque**2) * m.opt.timestep
                    joule_energy_step = hip_joule + knee_joule
                    
                    # Total energy (mechanical + joule)
                    total_energy_step = mech_energy_step 
                    
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
def get_field(data, *keys):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing required key(s): {keys}")


def get_motor_continuous_torque(config_path, motor_name):
    with open(config_path, "r") as f:
        data = json.load(f)

    motors = data.get("Motors", {})
    if motor_name not in motors:
        raise ValueError(f"Motor {motor_name} not found in {config_path}")

    motor = motors[motor_name]
    kv = motor["Kv"]
    max_continuous_current = motor["maxContinuousCurrent"]

    return (9.55 / kv) * max_continuous_current


case_key = CASE.strip().upper()
if case_key == "NOMINAL":
    case_key = "NOMINAL"
elif case_key not in CASE_CHOICES:
    raise ValueError("Invalid case. Choose A, B, C, or Nominal.")

case_label = "Nominal" if case_key == "NOMINAL" else f"Case_{case_key}"

results_json = CASE_CHOICES[case_key]["results_json"]
with open(results_json, "r") as f:
    results_data = json.load(f)

secondary = results_data.get("secondary")
if secondary is None:
    raise ValueError("No secondary entry found in JSON; rerun extract_results.py or relax matching.")

unique_id = get_field(secondary, "Unique id", "unique_id", "Unique Id")
xml_path = os.path.join(CASE_CHOICES[case_key]["xml_dir"], f"{unique_id}.xml")

ik_height = -float(get_field(secondary, "ik_height", "IK height", "ik_height (m)"))
thigh_length = float(get_field(secondary, "Thigh", "thigh_length"))
calf_length = float(get_field(secondary, "Calf", "calf_length"))
hip_offset = float(get_field(secondary, "Torso distance", "torso_distance")) * 0.5
efficiency_left = float(get_field(secondary, "Efficiency left", "efficiency_left"))
efficiency_right = float(get_field(secondary, "Efficiency right", "efficiency_right"))
ori_l = float(get_field(secondary, "ori_l", "ori_l (m)"))
ori_theta = float(get_field(secondary, "ori_theta", "ori_theta (rad)"))

motor_left_name = get_field(secondary, "Hip left motor", "motor_left_name", "motor_left")
motor_right_name = get_field(secondary, "Hip right motor", "motor_right_name", "motor_right")
gear_left_ratio = float(get_field(secondary, "Hip left ratio", "gear_left_ratio", "gear_ratio_left"))
gear_right_ratio = float(get_field(secondary, "Hip right ratio", "gear_right_ratio", "gear_ratio_right"))

config_path = "/home/stochlab/repo/optimal-design-legged-robots/COMPAct/config_files/config.json"
motor_left_key = f"Motor{motor_left_name}_framed"
motor_right_key = f"Motor{motor_right_name}_framed"
hip1_peak_torque = get_motor_continuous_torque(config_path, motor_left_key) * gear_left_ratio
hip2_peak_torque = get_motor_continuous_torque(config_path, motor_right_key) * gear_right_ratio

# Spring-damper-torsion gains
# [linear_kp, linear_kd, rotational_kp]
action = np.array([
    float(get_field(secondary, "ac1", "ac1 (kp)")),
    float(get_field(secondary, "ac2", "ac2 (kd)")),
    float(get_field(secondary, "ac3", "ac3 (kt)")),
])






results = run(
    xml_path,
    action,
    ik_value=ik_height,
    hip1_peak_torque=hip1_peak_torque,
    hip2_peak_torque=hip2_peak_torque,
    thigh_length=thigh_length,
    calf_length=calf_length,
    hip_offset=hip_offset,
    efficiency_left=efficiency_left,
    efficiency_right=efficiency_right,
    ori_l=ori_l,
    ori_theta=ori_theta,
    case_label=case_label,
)

print("Best Jump Height: {:.4f} m".format(results[0]))
print("Best Jump Forward Velocity: {:.4f} m/s".format(results[1]))
print("Best Jump Distance: {:.4f} m".format(results[2]))
print("Best Jump Energy: {:.4f} J".format(results[3]))
print("Best Jump Duration: {:.4f} s".format(results[4]))
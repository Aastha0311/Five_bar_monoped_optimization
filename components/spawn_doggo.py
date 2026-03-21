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

def run(xml_path, action, ik_value, hip1_peak_torque, hip2_peak_torque, thigh_length, calf_length, hip_offset, efficiency_left, efficiency_right):
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)

    m.opt.iterations = 500  
    m.opt.tolerance = 1e-10

    # record_video = True
    # video_filename = "5bar_best_6337_20.mp4"
    # video_fps = 100
    # if record_video:
    #     renderer = mj.Renderer(m, width=1920, height=1080)
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
    #mj.mj_forward(m, d)
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
    kp = 100
    kd = 50
    
    # Joule heating constant
    kt = 0.0955
    max_steps = 2000
    step_counter = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            # while jump_count < 1 and step_counter < max_steps:
                step_counter += 1  # Stop after 1 jump
                step_start = time.time()
                t_elapsed = time.time() - start        
                if t_elapsed < 5:
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
                    d.ctrl[hip_left_actuator_id] = hip_left_torque
                    d.ctrl[hip_right_actuator_id] = hip_right_torque
                    print(hip_left_torque, hip_right_torque)
                    print(d.qpos[slide_z_dof])
                    print(mj.mj_getTotalmass(m))
                    mj.mj_step(m, d)
                    viewer.sync()

xml_path = '/home/stochlab/repo/optimal-design-legged-robots/xmls/design_xmls/8778abeb.xml'
action = None
ik_value = -0.4203276682254768
hip1_peak_torque = 4.000092005818691*2.292
hip2_peak_torque = 9.04999755345784*2.304
thigh_length = 0.3046289869522943
calf_length = 0.190030742101534
hip_offset = 0.0500076970975257*0.5
efficiency_left = 0.963
efficiency_right = 0.939
run(xml_path, action, ik_value, hip1_peak_torque, hip2_peak_torque, thigh_length, calf_length, hip_offset, efficiency_left, efficiency_right)

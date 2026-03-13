import numpy as np
import mujoco as mj
import time
#import len_rad_id as lri
#import vmc_roc_action as vmc_ra
import ik as ik
import random
#import vmc_roc_angle as vmc_ra_angle
import pandas as pd
import glfw
from scipy.interpolate import interp1d
import vmc_roc_planar as vmc_rp
import mujoco.viewer as view
# Set fixed random seed for deterministic behavior
np.random.seed(0)
random.seed(0)

def run(xml_path, action, ik_value, hip_peak_torque, knee_peak_torque):
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)
    #viewer = view.launch_passive(m, d)
    af = pd.read_csv('/home/stochlab/repo/Aastha_Coopt_Monoped/Monoped-optimization/Post_Humanoids/data/comb_mass_torque.csv')

# Sort by gear ratio to avoid interpolation issues
    af = af.sort_values(by='gear_ratio')

    # Extract gear ratio and mass
    gear_ratios = af['gear_ratio'].values
    act_masses = af['mass'].values

    # Create a linear interpolation function without extrapolation
    act_interp_func = interp1d(
        gear_ratios,
        act_masses,
        kind='linear',
        bounds_error=True  # This raises an error if input is out of bounds
    )

    # hip_actuator_mass = act_interp_func(4.000146190606526)
    # knee_actuator_mass = act_interp_func(5.999997890823783)
    
    hip_actuator_mass = act_interp_func(6)
    knee_actuator_mass = act_interp_func(6)
    
    m.opt.iterations = 500  
    m.opt.tolerance = 1e-10
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
    l1n = 0.4
    l2n = 0.4
    dft = pd.read_csv('/home/stochlab/repo/Aastha_Coopt_Monoped/Monoped-optimization/Post_Humanoids/data/thigh_03_05.csv')

# Sort by gear ratio to avoid interpolation issues
    dft = dft.sort_values(by='Link Length (mm)')

    # Extract gear ratio and mass
    thigh_lengths = dft['Link Length (mm)'].values/1000
    thigh_masses = dft['Calculated Mass (kg)'].values

    # Create a linear interpolation function without extrapolation
    thigh_interp_func = interp1d(
        thigh_lengths,
        thigh_masses,
        kind='linear',
        bounds_error=True  # This raises an error if input is out of bounds
    )


    dfc = pd.read_csv('/home/stochlab/repo/Aastha_Coopt_Monoped/Monoped-optimization/Post_Humanoids/data/calf_03_05.csv')

    # Sort by gear ratio to avoid interpolation issues
    dfc = dfc.sort_values(by='Link Length (mm)')

    # Extract gear ratio and mass
    calf_lengths = dfc['Link Length (mm)'].values/1000
    calf_masses = dfc['Calculated Mass (kg)'].values

    # Create a linear interpolation function without extrapolation
    calf_interp_func = interp1d(
        calf_lengths,
        calf_masses,
        kind='linear',
        bounds_error=True  # This raises an error if input is out of bounds
    )

    thigh_link_mass = thigh_interp_func(l1n) 
    calf_link_mass = calf_interp_func(l2n)
    #mj.mj_setTotalmass(m, hip_actuator_mass + knee_actuator_mass + thigh_link_mass + calf_link_mass)
    #print(hip_actuator_mass+ knee_actuator_mass+ thigh_link_mass+ calf_link_mass)
    #exit()

    theta1, theta2 = ik.inverse_kinematics(0, ik_value, l1n, l2n)
    omega1_store = []
    omega2_store = []
    v0_store = []
    theta1_store = []
    theta2_store = []
    base_store = []
    torque1_store = []
    torque2_store = []
    time_store = [] 
    store_com = []   
    store_com_vel = []
    base_x_store = []
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
    
    #add viewer if you want to see the simulation

    
    while time.time() - start < 25 and jump_count < 3:  # Stop after 3 jumps
        step_start = time.time()
        t_elapsed = time.time() - start        
        if t_elapsed < 0:
            hip_error = 0 - d.qpos[2]
            knee_error = 0 - d.qpos[3]
            hip_torque = 100 * hip_error - 10 * d.qvel[2]
            knee_torque = 100 * knee_error - 10 * d.qvel[3]
        else:
            controller = vmc_rp.Controller(xml_path, m, d, theta1, theta2)
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
                    current_jump_start_x = d.qpos[0]
                    current_jump_start_time = t_elapsed
                
                elif jump_started and jump_phase == "air" and current_base_vel < 0:
                    # Landing phase: negative velocity while landing
                    jump_phase = "landing"
                
                elif jump_started and jump_phase == "landing" and current_base_vel >= 0:
                    # Jump completed, record results
                    jump_count += 1
                    current_jump_x_end = d.qpos[0]
                    jump_distance = current_jump_x_end - current_jump_start_x
                    jump_duration = t_elapsed - current_jump_start_time
                    euclidean_distance = np.sqrt(jump_distance**2 + current_jump_max_z**2)
                    jump_results.append((current_jump_max_z, current_jump_mech_energy, current_jump_joule_energy, current_jump_total_energy, 
                                        jump_distance, current_jump_x_end, jump_duration, euclidean_distance))
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
                    controller.lin_sd_var(action)
                    controller.rot_sd_var(action)
                    controller.linear_spring_force(action)
                    controller.linear_damper_force(action)
                    controller.torsional_spring_force(action)
                    controller.total_linear_force(action)
                    controller.fz_by_ee(action)
                    controller.fx_by_ee(action)
                    controller.force_applied_ground(action)
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
        
        d.ctrl[hip_actuator_id] = hip_torque
        d.ctrl[knee_actuator_id] = knee_torque

        
        # Calculate energy for the current jump (starts from ground contact with positive velocity)
        if jump_started and t_elapsed >= 0:
            # Mechanical energy (work done by actuators)
            ha = m.opt.timestep * hip_torque * d.qvel[2]            
            ka = m.opt.timestep * knee_torque * d.qvel[3]
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
            total_energy_step = mech_energy_step + joule_energy_step
            
            # Update jump energy accumulators
            current_jump_mech_energy += mech_energy_step
            current_jump_joule_energy += joule_energy_step
            current_jump_total_energy += total_energy_step
            
            # Also track total energy for all jumps
            hip_energy += ha
            knee_energy += ka
            total_energy += mech_energy_step
        
        # Step simulation
        omega1_store.append(d.qvel[2])
        base_store.append(d.qpos[1])
        base_x_store.append(d.qpos[0])
        torque1_store.append(hip_torque)
        torque2_store.append(knee_torque)
        theta1_store.append(d.qpos[2])
        theta2_store.append(d.qpos[3])
        omega2_store.append(d.qvel[3])
        v0_store.append(d.qvel[1])
        time_store.append(t_elapsed)
        com_z = d.subtree_com[1][2]
        store_com.append(com_z)
        
        mj.mj_step(m, d)
                # ---------------- Render frame ---------------- #
        #viewer.sync()
        # ----------------------------------------------- #

        masses = m.body_mass
        velocities = d.cvel[:, :3]   # linear part of each body's CoM velocity (Nx3)

        total_mass = np.sum(masses)
        com_vel = np.sum(masses[:, None] * velocities, axis=0) / total_mass
        com_vel_z = com_vel[0]

        store_com_vel.append(com_vel_z)
        
        # Respect simulation timestep
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    # Record the final jump if simulation ends during a jump
    if jump_started and jump_count < 3:
        jump_count += 1
        current_jump_x_end = d.qpos[0]
        jump_distance = current_jump_x_end - current_jump_start_x
        jump_duration = t_elapsed - current_jump_start_time
        euclidean_distance = np.sqrt(jump_distance**2 + current_jump_max_z**2)
        jump_results.append((current_jump_max_z, current_jump_mech_energy, current_jump_joule_energy, current_jump_total_energy, 
                            jump_distance, current_jump_x_end, jump_duration, euclidean_distance))
        max_height_after_control = max(max_height_after_control, current_jump_max_z)

    # Find the jump with maximum Euclidean distance (sqrt(x² + z²))
    if jump_results:
        max_euclidean_jump = max(jump_results, key=lambda x: x[7])  # index 7 is euclidean_distance
        (max_euclidean_height, max_euclidean_mech_energy, max_euclidean_joule_energy, max_euclidean_total_energy, 
         max_euclidean_distance, max_euclidean_x_end, max_euclidean_duration, max_euclidean_value) = max_euclidean_jump
    else:
        (max_euclidean_height, max_euclidean_mech_energy, max_euclidean_joule_energy, max_euclidean_total_energy, 
         max_euclidean_distance, max_euclidean_x_end, max_euclidean_duration, max_euclidean_value) = (-1e12, 1e12, 1e12, 1e12, 0, 0, 0, 0)
    
    # Also find the jump with max height for backward compatibility
    max_jump = max(jump_results, key=lambda x: x[0], default=(-1e12, 1e12, 1e12, 1e12, 0, 0, 0, 0))
    (max_jump_height, max_jump_mech_energy, max_jump_joule_energy, max_jump_total_energy, 
     max_jump_distance, max_jump_x_end, max_jump_duration, max_jump_euclidean) = max_jump
    
    #command to get total mass
    print(mj.mj_getTotalmass(m))

    return (max_height_after_control, jump_count, total_energy, hip_energy, knee_energy,
            max_jump_height, max_jump_mech_energy, max_jump_joule_energy, max_jump_total_energy, 
            max_jump_distance, max_jump_x_end, max_jump_duration,
            max_euclidean_height, max_euclidean_mech_energy, max_euclidean_joule_energy, max_euclidean_total_energy,
            max_euclidean_distance, max_euclidean_x_end, max_euclidean_duration, max_euclidean_value,
            jump_results, omega2_store, omega1_store, v0_store, theta1_store, theta2_store,
            base_store, torque1_store, torque2_store, time_store, store_com, store_com_vel, 
            base_x_store, current_x_max)

# Define Action & XML Path
action = np.array([100,5,10])
xml_path = '/home/stochlab/repo/Aastha_Coopt_Monoped/Monoped-optimization/Post_Humanoids/xmls/main xmls/gs_test.xml'
(max_height_after_control, jump_count, total_energy, hip_energy, knee_energy,
 max_jump_height, max_jump_mech_energy, max_jump_joule_energy, max_jump_total_energy, 
 max_jump_distance, max_jump_x_end, max_jump_duration,
 max_euclidean_height, max_euclidean_mech_energy, max_euclidean_joule_energy, max_euclidean_total_energy,
 max_euclidean_distance, max_euclidean_x_end, max_euclidean_duration, max_euclidean_value,
 jump_results, omega2_store, omega1_store, v0_store, theta1_store, theta2_store,
 base_store, torque1_store, torque2_store, time_store, store_com, store_com_vel, 
 base_x_store, current_x_max) = run(xml_path, action,-0.5, 6*5.5, 6*5.5*1.3636)

data = {
    "omega2_store": omega2_store,
    "omega1_store": omega1_store,
    "v0_store": v0_store,
    "theta1_store": theta1_store,
    "theta2_store": theta2_store,
    "base_store": base_store,
    "torque1_store": torque1_store,
    "torque2_store": torque2_store,
    "time_store": time_store,
    "store_com": store_com,
    "store_com_vel": store_com_vel, 
    "base_x_store": base_x_store
}

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("/home/stochlab/repo/Aastha_Coopt_Monoped/Monoped-optimization/Post_Humanoids/result_csvs/planar/nominal_planar_regenerative.csv", index=False)

print("All jumps results:")
for i, jump in enumerate(jump_results):
    height, mech_energy, joule_energy, total_energy, distance, x_end, duration, euclidean = jump
    print(f"Jump {i+1}: Height={height:.3f}m, Mech={mech_energy:.3f}J, Joule={joule_energy:.3f}J, Total={total_energy:.3f}J, "
          f"Distance={distance:.3f}m, X_end={x_end:.3f}m, Duration={duration:.3f}s, Euclidean={euclidean:.3f}m")

print("\nJump with maximum Euclidean distance (sqrt(x² + z²)):")
print(f"Height: {max_euclidean_height:.3f}m")
print(f"Mechanical Energy: {max_euclidean_mech_energy:.3f}J")
print(f"Joule Heating Energy: {max_euclidean_joule_energy:.3f}J")
print(f"Total Energy: {max_euclidean_total_energy:.3f}J")
print(f"Distance: {max_euclidean_distance:.3f}m")
print(f"X_end position: {max_euclidean_x_end:.3f}m")
print(f"Duration: {max_euclidean_duration:.3f}s")
print(f"Euclidean distance: {max_euclidean_value:.3f}m")

height_cost = 100*np.exp(-(0.5+max_height_after_control))
final_cost = 0.65*(height_cost) + 0.35*(max_jump_total_energy)  # Using total energy (mech + joule) for cost calculation
print(f"\nFinal cost: {final_cost}")
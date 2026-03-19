import csv
import cma
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter
#import rcp
import os
import sys
import multiprocessing
import uuid
from datetime import datetime
#from Post_Humanoids.controller_codes import final_rcp as rcp
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import sys, os
#sys.path.append("/home/stochlab/repo/Aastha_Coopt_Monoped/Monoped-optimization")
import old_rcp_2bar_wovideo as rcp
import json 
# Define the coefficient sets
coefficient_sets = []
for first_coeff in np.arange(0.3, 0.9, 0.05):  # 0.4 to 0.8 with step 0.05
    second_coeff = 1.0 - first_coeff
    coefficient_sets.append((first_coeff, second_coeff))

seed_list = np.linspace(5, 5, num=1)  # Modify this list for desired seeds
num_seeds = len(seed_list)

# Main loop for coefficient sets
for coeff_set in coefficient_sets:
    coeff1, coeff2 = coeff_set
    coeff_str = f"{int(coeff1*100):03d}_{int(coeff2*100):03d}"
    
    print(f"\n\n========== Running optimization for COEFFICIENTS = {coeff1:.2f}, {coeff2:.2f} ==========\n")
    
    for seed in seed_list:
        print(f"\n\n========== Running optimization for SEED = {seed} ==========\n")

        xml_template = "/home/stochlab/repo/optimal-design-legged-robots/xmls/stoch3.xml"
          
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Update filenames to include coefficient values
        best_results_file = f"/home/stochlab/repo/optimal-design-legged-robots/results/planar/dist/stoch3/best_dist_25_{coeff_str}_{date_str}_{seed}.csv"
        all_samples_file = f"/home/stochlab/repo/optimal-design-legged-robots/results/planar/dist/stoch3/all_dist_25_{coeff_str}_{date_str}_{seed}.csv"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(best_results_file), exist_ok=True)
        os.makedirs(os.path.dirname(all_samples_file), exist_ok=True)
        
        original_bounds = np.array([            
            [0.3, 0.6], #ik height
            [50, 1000],    # Controller Param 1
            [0, 10], 
            [10, 50]# Controller Param        # Controller Param 6
        ])
        import pandas as pd

        def motor_index_to_name(x):
            """
            Maps a numeric input to one of six motor names.
            """

            # round to nearest integer
            idx = int(round(x))

            # clamp to range 1–6
            idx = max(1, min(6, idx))

            motor_map = {
                1: "U8",
                2: "U10",
                3: "U12",
                4: "MN8014",
                5: "VT8020",
                6: "MAD_M6C12"
            }

            return motor_map[idx]


        def get_motor_gearbox_properties(csv_path, motor_name, gear_ratio):
            """
            Returns mass and efficiency for a given motor and gear ratio.

            Parameters
            ----------
            csv_path : str
                Path to motor-gearbox CSV
            motor_name : str
                Motor name used in CSV column 'motor'
            gear_ratio : float
                Gear ratio (will be rounded to 1 decimal)

            Returns
            -------
            mass : float
            efficiency : float
            """

            # round ratio to one decimal
            ratio = round(gear_ratio, 1)

            df = pd.read_csv(csv_path)

            # filter motor
            df_motor = df[df["motor"] == motor_name]

            if df_motor.empty:
                raise ValueError(f"Motor {motor_name} not found in CSV")

            # filter ratio
            idx = (df_motor["target_ratio"] - ratio).abs().idxmin()
            row = df_motor.loc[idx]
            #row = df_motor[df_motor["target_ratio"] == ratio]

            if row.empty:
                raise ValueError(
                    f"Motor {motor_name} with ratio {ratio} not found in CSV"
                )

            # mass = row.iloc[0]["mass"]
            # efficiency = row.iloc[0]["efficiency"]
            mass = row["mass"]
            efficiency = row["efficiency"]
            #get gearbox from the row
            gearbox = row["gearbox"]

            #print(f"Selected Motor: {motor_name}, Gear Ratio: {ratio}, Mass: {mass}, Efficiency: {efficiency}")

            return mass, efficiency, gearbox
        # Helper Functions
        dft = pd.read_csv('/home/stochlab/repo/optimal-design-legged-robots/components/thigh_15_35.csv')

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
            bounds_error=False,
            fill_value="extrapolate"  # This raises an error if input is out of bounds
        )


        dfc = pd.read_csv('/home/stochlab/repo/optimal-design-legged-robots/components/calf_15_35.csv')

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
            bounds_error=False,
            fill_value="extrapolate"  # This raises an error if input is out of bounds
        )



        def normalize(params):
            return (params - original_bounds[:, 0]) / (original_bounds[:, 1] - original_bounds[:, 0])

        def denormalize(norm_params):
            norm_params = np.clip(norm_params, 0, 1)
            return norm_params * (original_bounds[:, 1] - original_bounds[:, 0]) + original_bounds[:, 0]

        import xml.etree.ElementTree as ET
        def inverse_kinematics(x, z, l1, l2, branch=1):
            c2 = (x**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)
            if abs(c2) > 1:
                raise ValueError("Target out of reach")
            if branch == 1:
                theta2 = np.arctan2(-np.sqrt(1 - c2**2), c2)
            else:   
                theta2 = np.arctan2(np.sqrt(1 - c2**2), c2)
            A = l1 + l2 * c2
            B = l2 * np.sin(theta2)
            theta1 = np.arctan2(A*x + B*z, x*B - A*z)
            return theta1, theta2
        
        import xml.etree.ElementTree as ET


        def modify_link_length_and_angle(xml_file, output_file, 
                                            thigh_length, calf_length,
                                            hip_peak_torque, knee_peak_torque,
                                            efficiency_hip, efficiency_knee,
                                            mass_hip_actuator, mass_knee_actuator,
                                            target_base_height):
                
                # Compute joint angles
                l1, l2 = thigh_length, calf_length
                #l1, l2 = 0.4, 0.4  # Default values for testing
                # hip_peak_torque = thigh_gear_ratio*5.5
                # knee_peak_torque = knee_gear_ratio*5.5*1.3636
                theta1, theta2 = inverse_kinematics(0, -target_base_height, l1, l2)   

                # Compute (x, z) vector for thigh and shank using rotation
                thigh_vec = [l1 * np.sin(theta1), 0, -l1 * np.cos(theta1)]
                knee_pos = thigh_vec
                shank_vec = [l2 * np.sin(theta1 + theta2), 0, 
                            -l2 * np.cos(theta1 + theta2)]

                tree = ET.parse(xml_file)
                root = tree.getroot()
                calf_offset = -0.025*np.cos(theta1 + theta2)
                calf_x_offset = 0.025*np.sin(theta1 + theta2)
                theta1d = theta1 * 180 / np.pi
                theta2d = theta2 * 180 / np.pi
                # hip_actuator_mass = act_interp_func(thigh_gear_ratio)
                # knee_actuator_mass = act_interp_func(knee_gear_ratio)
                torso_mass = mass_hip_actuator + mass_knee_actuator
                #torso_mass = 0.0
                # thigh_link_mass = 0
                # calf_link_mass = 0
                thigh_link_mass = thigh_interp_func(thigh_length) 
                calf_link_mass = calf_interp_func(calf_length)
                for worldbody in root.findall('worldbody'):
                    for torso in worldbody.findall('body'):
                        torso.set('pos', f"0 0 {target_base_height}")
                        for base in torso.findall('body'):
                            for geom in base.findall('geom'):
                                if torso_mass:
                                    geom.set('mass', str(torso_mass))
                            for thigh in base.findall('body'):
                                if thigh.get('name') == "link1":
                                    thigh.set('pos', "0 0 0")
                                    thigh.set('euler', f"0 {(np.pi*0.5)-theta1} 0")
                                    for geom in thigh.findall('geom'):
                                        if geom.get('name') == "thigh":                            
                                            geom.set('fromto', f"0 0 0 {l1} 0 0")
                                            if thigh_link_mass:
                                                geom.set('mass', str(thigh_link_mass))
                                    for joint in thigh.findall('joint'):                        
                                        joint.set('pos', "0 0 0")
                                for calf in thigh.findall('body'):
                                    if calf.get('name') == "link2":                        
                                        calf.set('pos', f"{l1} 0 0")
                                        calf.set('euler', f"0 {-theta2} 0")                        
                                        for geom in calf.findall('geom'):
                                            if geom.get('name') == "shank":                            
                                                geom.set('fromto', f"0 0 0 {l2} 0 0")                            
                                                if calf_link_mass:
                                                    geom.set('mass', str(calf_link_mass))                            
                                        for joint in calf.findall('joint'):                            
                                            joint.set('pos', "0 0 0")
                # Update ctrlrange if needed
                for motor in root.findall(".//motor"):
                    if motor.get("name") == "torque1" and hip_peak_torque:
                        motor.set("ctrlrange", f"-{efficiency_hip*hip_peak_torque} {efficiency_hip*hip_peak_torque}")
                    elif motor.get("name") == "torque2" and knee_peak_torque:
                        motor.set("ctrlrange", f"-{efficiency_knee*knee_peak_torque} {efficiency_knee*knee_peak_torque}")

                tree.write(output_file)

        import pandas as pd

        def get_continuous_torque(json_path, motor_name):
            """
            Returns continuous motor torque from motor JSON data.
            """

            with open(json_path, "r") as f:
                data = json.load(f)

            motors = data["Motors"]

            if motor_name not in motors:
                raise ValueError(f"Motor {motor_name} not found in JSON")

            motor = motors[motor_name]

            Kv = motor["Kv"]                       # rpm/V
            I_cont = motor["maxContinuousCurrent"] # A

            # torque constant
            Kt = 60 / (2 * np.pi * Kv)

            # continuous torque
            tau_cont = Kt * I_cont
            #print(f"Continuous torque for {motor_name}: {tau_cont:.3f} Nm")

            return tau_cont




        def process_action(action):
            action = action.copy()
            for i in range(len(action)):
                action[i] = np.round(action[i], 1)
            return action

        def round_to_nearest(x, base=0.005):
            return round(base * round(x / base), 3)

        
        def get_cost(params):

            try:

                params = denormalize(params)

                # thigh_length = params[0]
                # calf_length = params[1]
                # torso_distance = params[2]
                thigh_length = 0.297
                calf_length = 0.302
                torso_distance = 0.125
                ik_height = params[0]
                max_reach = thigh_length + calf_length
                min_reach = abs(thigh_length - calf_length)
                ik_height = np.clip(ik_height, min_reach, max_reach-0.01)
                # motor_left_name = motor_index_to_name(params[4])
                # motor_right_name = motor_index_to_name(params[5])
                motor_hip_index = 2
                motor_knee_index = 2
                motor_hip_name = motor_index_to_name(motor_hip_index)
                motor_knee_name = motor_index_to_name(motor_knee_index)
                gear_hip_ratio = 6.0
                gear_knee_ratio = 6.0

                # gear_left_ratio = params[6]
                # gear_right_ratio = params[7]

                controller_params = process_action(params[1:])

                mass_hip, efficiency_hip, gearbox_hip = get_motor_gearbox_properties(
                    "/home/stochlab/repo/optimal_gearbox_selection.csv",
                    motor_hip_name,
                    gear_hip_ratio
                )

                mass_knee, efficiency_knee, gearbox_knee = get_motor_gearbox_properties(
                    "/home/stochlab/repo/optimal_gearbox_selection.csv",
                    motor_knee_name,
                    gear_knee_ratio
                )

                tau_hip = get_continuous_torque(
                    "/home/stochlab/repo/optimal-design-legged-robots/COMPAct/config_files/config.json",
                    "Motor" + motor_hip_name + "_framed"
                ) * gear_hip_ratio

                tau_knee = get_continuous_torque(
                    "/home/stochlab/repo/optimal-design-legged-robots/COMPAct/config_files/config.json",
                    "Motor" + motor_knee_name + "_framed"
                ) * gear_knee_ratio

                unique_id = uuid.uuid4().hex[:8]

                modified_xml = f"/home/stochlab/repo/optimal-design-legged-robots/xmls/stoch3_xmls/{unique_id}.xml"

                modify_link_length_and_angle(
                    xml_file=xml_template,
                    output_file=modified_xml,
                    thigh_length=thigh_length,
                    calf_length=calf_length,
                    hip_peak_torque=tau_hip,
                    knee_peak_torque=tau_knee,
                    efficiency_hip=efficiency_hip,
                    efficiency_knee=efficiency_knee,
                    mass_hip_actuator=mass_hip,
                    mass_knee_actuator=mass_knee,
                    target_base_height=ik_height
                )
                # -------------------------
                # RUN SIMULATION 5 TIMES
                # -------------------------

                run_results = []

                for i in range(5):

                    result = rcp.run(
                        modified_xml,
                        controller_params,
                        -ik_height,
                        tau_hip,
                        tau_knee,
                        thigh_length,
                        calf_length,
                        efficiency_hip,
                        efficiency_knee
                    )

                    if result is None:
                        continue

                    best_height, best_x_vel, best_distance, best_energy, best_duration, jump_results = result

                    best_x_vel = abs(best_x_vel)
                    best_distance = abs(best_distance)

                    run_results.append({
                        "height": best_height,
                        "xvel": best_x_vel,
                        "distance": best_distance,
                        "energy": best_energy,
                        "duration": best_duration
                    })

                if len(run_results) == 0:
                    return 1e6

                # -------------------------
                # MODE DISTANCE SELECTION
                # -------------------------

                rounded_distances = [round(r["distance"], 2) for r in run_results]

                distance_counts = Counter(rounded_distances)
                mode_distance = distance_counts.most_common(1)[0][0]

                # pick first run matching the mode
                selected_run = None

                for r in run_results:
                    if round(r["distance"], 2) == mode_distance:
                        selected_run = r
                        break

                best_height = selected_run["height"]
                best_x_vel = selected_run["xvel"]
                best_distance = selected_run["distance"]
                best_energy = selected_run["energy"]

                # -------------------------
                # COST
                # -------------------------

                cost = coeff1 * (-best_distance * 25) + coeff2 * (best_energy)

                # -------------------------
                # SAVE ALL RUNS
                # -------------------------

                with open(all_samples_file, "a", newline="") as file:

                    writer = csv.writer(file)

                    for r in run_results:

                        writer.writerow([
                            thigh_length, calf_length,
                            motor_hip_name, motor_knee_name,
                            gear_hip_ratio, gear_knee_ratio,
                            gearbox_hip, gearbox_knee, efficiency_hip, efficiency_knee,
                            torso_distance,ik_height, r["xvel"], r["energy"],
                            r["height"], r["distance"],
                            unique_id
                        ] + list(controller_params))

                return cost

            except Exception as e:

                print("Simulation failed:", e)

                return 1e6
        
            

        # CMA-ES Optimization
        x0 = normalize(np.array([0.35, 550, 5, 30]))
        sigma0 = 0.1
        opts = cma.CMAOptions()
        opts.set({
            'maxiter': 1000, 'popsize': 8, 'seed': int(seed),
            'bounds': [np.zeros(4), np.ones(4)], 'verb_disp': 1000, 'verb_disp': 1})
            # 'tolfun': 0,
            # 'tolfunhist': 0,
            # 'tolx': 0,
            # 'tolstagnation': 1000,

            
                

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        with open(best_results_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["thigh_length", "calf_length", "motor_hip_name", "motor_knee_name", "gear_hip_ratio", "gear_knee_ratio", "gearbox_hip", "gearbox_knee", "torso_distance","ik_height", "best_index", "best_cost"] + [f"ac{i+1}" for i in range(3)])


        with open(all_samples_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Thigh", "Calf", "Hip motor", "Knee motor", "Hip ratio", "Knee ratio", "Gearbox hip", "Gearbox knee", "Efficiency hip", "Efficiency knee", "Torso distance", "ik_height", "Best X velocity", "Average energy", "Max height", "Max distance", "Unique id"] + [f"ac{i+1}" for i in range(3)])

        if __name__ == "__main__":

            pool = multiprocessing.Pool(processes=8)

            while not es.stop():

                solutions = es.ask()

                costs = pool.map(get_cost, solutions)

                #print("costs:", costs)

                best_index = np.argmin(costs)
                best_params = denormalize(solutions[best_index])
                best_cost = costs[best_index]

                # thigh_length = best_params[0]
                # calf_length = best_params[1]
                # torso_distance = best_params[2]
                # ik_height = best_params[3]
                thigh_length = 0.297
                calf_length = 0.302
                torso_distance = 0.125
                ik_height = best_params[0]
                motor_hip_index = 2
                motor_knee_index = 2
                motor_hip_name = motor_index_to_name(motor_hip_index)
                motor_knee_name = motor_index_to_name(motor_knee_index)
                gear_hip_ratio = 6.0
                gear_knee_ratio = 6.0
                # motor1_number = best_params[4]
                # motor2_number = best_params[5]

                # motor_left_name = motor_index_to_name(motor1_number)
                # motor_right_name = motor_index_to_name(motor2_number)

                # gear_left_ratio = best_params[6]
                # gear_right_ratio = best_params[7]

                controller_params = process_action(best_params[1:])

                mass_hip, efficiency_hip, gearbox_hip = get_motor_gearbox_properties(
                    "/home/stochlab/repo/optimal_gearbox_selection.csv",
                    motor_hip_name,
                    gear_hip_ratio
                )

                mass_knee, efficiency_knee, gearbox_knee = get_motor_gearbox_properties(
                    "/home/stochlab/repo/optimal_gearbox_selection.csv",
                    motor_knee_name,
                    gear_knee_ratio
                )

                with open(best_results_file, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        thigh_length, calf_length,
                        motor_hip_name, motor_knee_name,
                        gear_hip_ratio, gear_knee_ratio,
                        gearbox_hip, gearbox_knee,
                        torso_distance, ik_height,
                        best_index, best_cost
                    ] + list(controller_params))

                es.tell(solutions, costs)

            pool.close()
            pool.join()
            


        print(f"Joint optimization completed for seed {seed}.")
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        #store time taken in the end of best results file
        with open(best_results_file, "a", newline="") as file:
            writer = csv.writer(file)
            #subtract start time from end time
            end_time = datetime.now()
            # start_time = datetime.strptime(es.start_time, "%Y-%m-%d %H:%M:%S")
            # start_time = 
            # time_taken = (end_time - start_time).total_seconds()
            writer.writerow(["Time taken (seconds)"])
            writer.writerow([end_time.strftime("%Y-%m-%d %H:%M:%S")])
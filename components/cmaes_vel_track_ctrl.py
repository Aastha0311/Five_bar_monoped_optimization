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
import final_rcp_noslide_energy as rcp
import json 
# Define the coefficient sets
coefficient_sets = []
for first_coeff in np.arange(0.65, 0.70, 0.05):  # 0.4 to 0.8 with step 0.05
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

        xml_template = "/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar_base.xml"
          
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Update filenames to include coefficient values
        best_results_file = f"/home/stochlab/repo/optimal-design-legged-robots/results/planar/comb/best_comb_{coeff_str}_{date_str}_{seed}.csv"
        all_samples_file = f"/home/stochlab/repo/optimal-design-legged-robots/results/planar/comb/all_comb_{coeff_str}_{date_str}_{seed}.csv"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(best_results_file), exist_ok=True)
        os.makedirs(os.path.dirname(all_samples_file), exist_ok=True)
        
        original_bounds = np.array([
        [0.15,0.35],   # thigh
        [0.15,0.35],   # calf
        [0.05,0.15],   # torso

        [1,6],         # motor left
        [1,6],         # motor right

        [4,60],        # gear left
        [4,60],        # gear right
        [0.05,0.20],
        # controller v=2
        [50,500],
        [0,10],
        [10,50],

        # controller v=4
        [50,500],
        [0,10],
        [10,50],

        # controller v=6
        [50,500],
        [0,10],
        [10,50],

            # push duration
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

            print(f"Selected Motor: {motor_name}, Gear Ratio: {ratio}, Mass: {mass}, Efficiency: {efficiency}")

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


        def modify_5bar_xml(
                xml_file,
                output_file,
                base_height,
                l1,
                l2,
                torso_width,
                torque_left,
                torque_right,
                mass_left,
                mass_right,
                thigh_interp_func,
                calf_interp_func):

            tree = ET.parse(xml_file)
            root = tree.getroot()

            half_width = torso_width / 2

            # -----------------------------
            # Link masses from interpolation
            # -----------------------------
            thigh_mass = float(thigh_interp_func(l1))
            calf_mass = float(calf_interp_func(l2))
            print(f"Interpolated thigh mass for length {l1:.3f} m: {thigh_mass:.3f} kg")
            print(f"Interpolated calf mass for length {l2:.3f} m: {calf_mass:.3f} kg")
            # -----------------------------
            # Update torso width + mass
            # -----------------------------

            for body in root.findall(".//body[@name='root']"):
                pos = body.get("pos").split()
                pos[2] = str(base_height)
                body.set("pos", " ".join(pos))
            for geom in root.findall(".//geom[@name='torso']"):

                size = geom.get("size").split()
                size[0] = str(half_width)
                geom.set("size", " ".join(size))

                base_mass = 0.0
                total_mass = base_mass + mass_left + mass_right
                geom.set("mass", str(total_mass))

            # -----------------------------
            # Move hip joint locations
            # -----------------------------
            for body in root.findall(".//body[@name='l1_left']"):
                body.set("pos", f"{-half_width} 0 0")

            for body in root.findall(".//body[@name='l1_right']"):
                body.set("pos", f"{half_width} 0 0")

            # -----------------------------
            # Update thigh lengths + mass
            # -----------------------------
            for geom in root.findall(".//geom[@name='thigh_left']"):
                geom.set("fromto", f"0 0 0 {l1} 0 0")
                geom.set("mass", str(thigh_mass))

            for geom in root.findall(".//geom[@name='thigh_right']"):
                geom.set("fromto", f"0 0 0 {l1} 0 0")
                geom.set("mass", str(thigh_mass))

            # -----------------------------
            # Move knee joints
            # -----------------------------
            for body in root.findall(".//body[@name='l2_left']"):
                body.set("pos", f"{l1} 0 0")

            for body in root.findall(".//body[@name='l2_right']"):
                body.set("pos", f"{l1} 0 0")

            # -----------------------------
            # Update calf/shank lengths + mass
            # -----------------------------
            for geom in root.findall(".//geom[@name='shank_left']"):
                geom.set("fromto", f"0 0 0 {l2} 0 0")
                geom.set("mass", str(calf_mass))

            for geom in root.findall(".//geom[@name='shank_right']"):
                geom.set("fromto", f"0 0 0 {l2} 0 0")
                geom.set("mass", str(calf_mass))

            # -----------------------------
            # Move feet
            # -----------------------------
            for geom in root.findall(".//geom[@name='foot_left']"):
                geom.set("pos", f"{l2} 0 0")

            for geom in root.findall(".//geom[@name='foot_right']"):
                geom.set("pos", f"{l2} 0 0")

            # -----------------------------
            # Move coupler sites
            # -----------------------------
            for site in root.findall(".//site[@name='left_tip']"):
                site.set("pos", f"{l2} 0 0")

            for site in root.findall(".//site[@name='right_tip']"):
                site.set("pos", f"{l2} 0 0")

            # -----------------------------
            # Update torque limits
            # -----------------------------
            for motor in root.findall(".//motor[@name='motor_left']"):
                motor.set("ctrlrange", f"-{torque_left} {torque_left}")

            for motor in root.findall(".//motor[@name='motor_right']"):
                motor.set("ctrlrange", f"-{torque_right} {torque_right}")

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
            print(f"Continuous torque for {motor_name}: {tau_cont:.3f} Nm")

            return tau_cont




        def process_action(action):
            action = action.copy()
            for i in range(len(action)):
                action[i] = np.round(action[i], 1)
            return action

        def round_to_nearest(x, base=0.005):
            return round(base * round(x / base), 3)

        

        def get_cost(params):

            params = denormalize(params)

            # ---------------- DESIGN ----------------

            thigh_length = params[0]
            calf_length = params[1]
            torso_distance = params[2]

            motor_left_name = motor_index_to_name(params[3])
            motor_right_name = motor_index_to_name(params[4])

            gear_left_ratio = params[5]
            gear_right_ratio = params[6]

            push_duration = params[7]

            # ---------------- CONTROLLERS ----------------

            controller_v2 = process_action(params[8:11])
            controller_v4 = process_action(params[11:14])
            controller_v6 = process_action(params[14:17])

            # ---------------- MOTOR PROPERTIES ----------------

            mass_left, efficiency_left, gearbox_left = get_motor_gearbox_properties(
                "/home/stochlab/repo/optimal_gearbox_selection.csv",
                motor_left_name,
                gear_left_ratio
            )

            mass_right, efficiency_right, gearbox_right = get_motor_gearbox_properties(
                "/home/stochlab/repo/optimal_gearbox_selection.csv",
                motor_right_name,
                gear_right_ratio
            )

            tau_left = get_continuous_torque(
                "/home/stochlab/repo/optimal-design-legged-robots/COMPAct/config_files/config.json",
                "Motor" + motor_left_name + "_framed"
            ) * gear_left_ratio

            tau_right = get_continuous_torque(
                "/home/stochlab/repo/optimal-design-legged-robots/COMPAct/config_files/config.json",
                "Motor" + motor_right_name + "_framed"
            ) * gear_right_ratio

            unique_id = uuid.uuid4().hex[:8]

            modified_xml = f"/home/stochlab/repo/optimal-design-legged-robots/xmls/design_xmls/{unique_id}.xml"

            modify_5bar_xml(
                xml_template,
                modified_xml,
                0.3,
                thigh_length,
                calf_length,
                torso_distance,
                tau_left,
                tau_right,
                mass_left,
                mass_right,
                thigh_interp_func,
                calf_interp_func
            )

            # ---------------- MULTI-VELOCITY EVALUATION ----------------

            targets = [5,10,15]
            controllers = [controller_v2, controller_v4, controller_v6]

            velocities = []
            tracking_errors = []
            energies = []

            total_tracking_error = 0
            total_energy = 0

            for v_target, ctrl in zip(targets, controllers):

                result = rcp.run(
                    modified_xml,
                    ctrl,
                    -0.3,
                    tau_left,
                    tau_right,
                    thigh_length,
                    calf_length,
                    torso_distance*0.5,
                    push_duration,
                    efficiency_left,
                    efficiency_right
                )

                if result is None:
                    return 1e6

                best_height, best_x_vel, best_distance, best_energy, best_duration, jump_results = result

                actual_vel = abs(best_x_vel)

                # -------- NORMALIZED TRACKING ERROR --------

                tracking_error = ((actual_vel - v_target) / v_target)**2

                velocities.append(actual_vel)
                tracking_errors.append(tracking_error)
                energies.append(best_energy)

                total_tracking_error += tracking_error
                total_energy += best_energy

            # ---------------- COST ----------------

            cost = total_tracking_error + 0.05 * total_energy

            # ---------------- SAVE (UNCHANGED STRUCTURE) ----------------

            with open(all_samples_file, "a", newline="") as file:

                writer = csv.writer(file)

                writer.writerow([

                thigh_length,
                calf_length,
                motor_left_name,
                motor_right_name,
                gear_left_ratio,
                gear_right_ratio,
                gearbox_left,
                gearbox_right,
                torso_distance,
                push_duration,

                velocities[0], tracking_errors[0], energies[0],
                velocities[1], tracking_errors[1], energies[1],
                velocities[2], tracking_errors[2], energies[2],

                total_tracking_error,
                total_energy,

                unique_id

                ] + list(controller_v2) + list(controller_v4) + list(controller_v6))

                file.flush()

            return cost

            

        # CMA-ES Optimization
        x0 = normalize(np.array([

# DESIGN
0.25,   # thigh
0.25,   # calf
0.10,   # torso

3,      # motor left
4,      # motor right

10.0,   # gear left
10.0,   # gear right

0.125,  # push duration

# controller v2
200,
5,
30,

# controller v4
200,
5,
30,

# controller v6
200,
5,
30

]))
        sigma0 = 0.1
        opts = cma.CMAOptions()
        opts.set({
            'maxiter': 200, 'popsize': 8, 'seed': int(seed),
            'bounds': [np.zeros(len(original_bounds)), np.ones(len(original_bounds))], 'verb_disp': 1000
        })

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        with open(best_results_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
"Thigh",
"Calf",
"Hip left motor",
"Hip right motor",
"Hip left ratio",
"Hip right ratio",
"Gearbox left",
"Gearbox right",
"Torso distance",
"Push duration",

# velocity 2 results
"v2_actual_vel",
"v2_tracking_error",
"v2_mech_energy",

# velocity 4 results
"v4_actual_vel",
"v4_tracking_error",
"v4_mech_energy",

# velocity 6 results
"v6_actual_vel",
"v6_tracking_error",
"v6_mech_energy",

# totals
"total_tracking_error",
"total_energy",

"Unique id",

# controllers
"v2_ac1","v2_ac2","v2_ac3",
"v4_ac1","v4_ac2","v4_ac3",
"v6_ac1","v6_ac2","v6_ac3"
])


        with open(all_samples_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
"Thigh","Calf",
"Hip left motor","Hip right motor",
"Hip left ratio","Hip right ratio",
"Gearbox left","Gearbox right",
"Torso distance","Push duration",

"v2_actual","v2_error","v2_energy",
"v4_actual","v4_error","v4_energy",
"v6_actual","v6_error","v6_energy",

"total_tracking_error",
"total_energy",

"unique_id",

"v2_c1","v2_c2","v2_c3",
"v4_c1","v4_c2","v4_c3",
"v6_c1","v6_c2","v6_c3"
])


        while not es.stop():

            solutions = es.ask()

            with multiprocessing.Pool(processes=8) as pool:
                costs = pool.map(get_cost, solutions)

            #costs = [get_cost(s) for s in solutions]

            best_index = np.argmin(costs)

            best_params = denormalize(solutions[best_index])
            best_cost = costs[best_index]

            # ---------------- DESIGN ----------------

            thigh_length = best_params[0]
            calf_length = best_params[1]
            torso_distance = best_params[2]

            motor1_number = best_params[3]
            motor2_number = best_params[4]

            motor_left_name = motor_index_to_name(motor1_number)
            motor_right_name = motor_index_to_name(motor2_number)

            gear_left_ratio = best_params[5]
            gear_right_ratio = best_params[6]

            push_duration = best_params[7]

            # ---------------- CONTROLLERS ----------------

            controller_v2 = process_action(best_params[8:11])
            controller_v4 = process_action(best_params[11:14])
            controller_v6 = process_action(best_params[14:17])

            # ---------------- MOTOR PROPERTIES ----------------

            mass_left, efficiency_left, gearbox_left = get_motor_gearbox_properties(
                "/home/stochlab/repo/optimal_gearbox_selection.csv",
                motor_left_name,
                gear_left_ratio
            )

            mass_right, efficiency_right, gearbox_right = get_motor_gearbox_properties(
                "/home/stochlab/repo/optimal_gearbox_selection.csv",
                motor_right_name,
                gear_right_ratio
            )

            # ---------------- SAVE BEST ----------------

            with open(best_results_file, "a", newline="") as file:

                writer = csv.writer(file)

                writer.writerow([

                thigh_length,
                calf_length,
                motor_left_name,
                motor_right_name,
                gear_left_ratio,
                gear_right_ratio,
                gearbox_left,
                gearbox_right,
                torso_distance,
                push_duration,

                best_index,
                best_cost

                ] + list(controller_v2) + list(controller_v4) + list(controller_v6))

            es.tell(solutions, costs)
            


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
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
import json
from scipy.interpolate import interp1d
import pandas as pd


def inverse_kinematics(x, z, l1, l2, branch=1):
    c2 = (x**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if abs(c2) > 1:
        raise ValueError("Target out of reach")
    if branch == 1:
        theta2 = np.arctan2(-np.sqrt(1 - c2**2), c2)
    else:   
        theta2 = np.arctan2(np.sqrt(1 - c2**2), c2)
    #theta1 = np.arctan2(z, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    A = l1 + l2 * c2
    B = l2 * np.sin(theta2)
    theta1 = np.arctan2(A*x + B*z, x*B - A*z)

    print(f"theta1: {theta1}, theta2: {theta2}, theta1n: {theta1}")
    return theta1, theta2

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
    bounds_error=True  # This raises an error if input is out of bounds
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
    bounds_error=True  # This raises an error if input is out of bounds
)



def modify_link_length_and_angle(xml_file, output_file, 
                                    thigh_length, calf_length,
                                    hip_peak_torque, knee_peak_torque,
                                    mass_hip_actuator, mass_knee_actuator,
                                    target_base_height=-0.40):
        
        # Compute joint angles
        l1, l2 = thigh_length, calf_length
        #l1, l2 = 0.4, 0.4  # Default values for testing
        # hip_peak_torque = thigh_gear_ratio*5.5
        # knee_peak_torque = knee_gear_ratio*5.5*1.3636
        theta1, theta2 = inverse_kinematics(0, target_base_height, l1, l2)   

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
                torso.set('pos', f"0 0 {-target_base_height}")
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
                motor.set("ctrlrange", f"-{hip_peak_torque} {hip_peak_torque}")
            elif motor.get("name") == "torque2" and knee_peak_torque:
                motor.set("ctrlrange", f"-{knee_peak_torque*1.3636} {knee_peak_torque*1.3636}")

        tree.write(output_file)


l1n = 0.297
l2n= 0.302
gt= 6.0
gk= 6.0
motor_left_index = 2
motor_right_index = 2
motor_left_name = motor_index_to_name(motor_left_index)
motor_right_name = motor_index_to_name(motor_right_index)
mass_hip_actuator, _, _ = get_motor_gearbox_properties('/home/stochlab/repo/optimal_gearbox_selection.csv', motor_left_name, gt)
mass_knee_actuator, _, _ = get_motor_gearbox_properties('/home/stochlab/repo/optimal_gearbox_selection.csv', motor_right_name, gk)  
tau_hip = get_continuous_torque(
                    "/home/stochlab/repo/optimal-design-legged-robots/COMPAct/config_files/config.json",
                    "Motor" + motor_left_name + "_framed"
                ) * gt

tau_knee = get_continuous_torque(
                    "/home/stochlab/repo/optimal-design-legged-robots/COMPAct/config_files/config.json",
                    "Motor" + motor_right_name + "_framed"
                ) * gk

print(f"l1n: {l1n}, l2n: {l2n}")   
modify_link_length_and_angle(
    xml_file="/home/stochlab/repo/optimal-design-legged-robots/xmls/modified_robot_planar.xml", output_file= "/home/stochlab/repo/optimal-design-legged-robots/xmls/stoch3_40.xml", thigh_length = l1n, calf_length = l2n, hip_peak_torque = tau_hip, knee_peak_torque = tau_knee, mass_hip_actuator = mass_hip_actuator, mass_knee_actuator = mass_knee_actuator)
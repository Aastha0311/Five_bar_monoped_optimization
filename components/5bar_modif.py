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
#import old_rcp_5bar_wovideo as rcp
import json 

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
        efficiency_left,
        efficiency_right,
        mass_left,
        mass_right,
        thigh_interp,
        calf_interp):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    half_width = torso_width / 2

    # -----------------------------
    # Link masses from interpolation
    # -----------------------------
    thigh_mass = float(calf_interp(l1))
    calf_mass = float(calf_interp(l2))
    # print(f"Interpolated thigh mass for length {l1:.3f} m: {thigh_mass:.3f} kg")
    # print(f"Interpolated calf mass for length {l2:.3f} m: {calf_mass:.3f} kg")
    # -----------------------------
    # Update torso width + mass
    # -----------------------------

    for body in root.findall(".//body[@name='root']"):
        pos = body.get("pos").split()
        pos[2] = str(base_height)
        body.set("pos", " ".join(pos))
    for geom in root.findall(".//geom[@name='torso']"):

        size = geom.get("size").split()
        size[0] = str(torso_width)
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
        motor.set("ctrlrange", f"-{efficiency_left*torque_left} {efficiency_left*torque_left}")

    for motor in root.findall(".//motor[@name='motor_right']"):
        motor.set("ctrlrange", f"-{efficiency_right*torque_right} {efficiency_right*torque_right}")

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

# xml_path="/home/stochlab/repo/optimal-design-legged-robots/xmls/design_xmls/8778abeb.xml"

# modified_xml_path="/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar_baseline.xml"

ik_height = 0.45
thigh_length = 0.297
calf_length = 0.302
torso_distance = 0.125
motor_left_index = 5
motor_right_index = 2

motor_left_name = motor_index_to_name(motor_left_index)
motor_right_name = motor_index_to_name(motor_right_index)

gear_left_ratio = 6.0
gear_right_ratio = 6.0



mass_left_actuator, efficiency_left_actuator, gearbox_left = get_motor_gearbox_properties(
    "/home/stochlab/repo/optimal_gearbox_selection.csv",
    motor_left_name,
    gear_left_ratio
)

mass_right_actuator, efficiency_right_actuator, gearbox_right = get_motor_gearbox_properties(
    "/home/stochlab/repo/optimal_gearbox_selection.csv",
    motor_right_name,
    gear_right_ratio
)
print(efficiency_left_actuator, efficiency_right_actuator)

tau_left = get_continuous_torque(
    "/home/stochlab/repo/optimal-design-legged-robots/COMPAct/config_files/config.json",
    "Motor" + motor_left_name + "_framed"
) * gear_left_ratio

tau_right = get_continuous_torque(
    "/home/stochlab/repo/optimal-design-legged-robots/COMPAct/config_files/config.json",
    "Motor" + motor_right_name + "_framed"
) * gear_right_ratio

unique_id = uuid.uuid4().hex[:8]

modify_5bar_xml(
    xml_file="/home/stochlab/repo/optimal-design-legged-robots/xmls/design_xmls/8778abeb.xml",
    output_file="/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar_baseline.xml",
    base_height=ik_height,
    l1=thigh_length,
    l2=calf_length,
    torso_width=torso_distance,
    torque_left=tau_left,
    torque_right=tau_right,
    efficiency_left=efficiency_left_actuator,
    efficiency_right=efficiency_right_actuator,
    mass_left=mass_left_actuator,
    mass_right=mass_right_actuator,
    thigh_interp=thigh_interp_func,
    calf_interp=calf_interp_func
)


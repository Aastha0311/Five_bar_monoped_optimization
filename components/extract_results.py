import json
import pandas as pd
import numpy as np

# File paths
primary_csv = '/home/stochlab/repo/optimal-design-legged-robots/results/5bar_planar_2/best_dist_20_055_044_2026-03-24_02-08-14_5.0.csv'
secondary_csv = '/home/stochlab/repo/optimal-design-legged-robots/results/5bar_planar_2/all_dist_20_055_044_2026-03-24_02-08-14_5.0.csv'
output_json = '/home/stochlab/repo/optimal-design-legged-robots/results/analysis/5bar_ll_2.json'

# Load primary CSV
df1 = pd.read_csv(primary_csv)
df2 = pd.read_csv(secondary_csv)
print(df1.columns)
print(df2.columns)

# Ensure required columns (use actual headers from the best file)
required_cols = list(df1.columns)
required_primary_keys = [
    "best_cost",
    "ac1",
    "ac2",
    "ac3",
    "thigh_length",
    "calf_length",
    "ori_l",
    "ori_theta",
]
missing_cols = [col for col in required_primary_keys if col not in df1.columns]
if missing_cols:
    raise ValueError(f"Primary CSV missing columns: {missing_cols}")

for col in required_primary_keys:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')

# # # Get row with lowest Best Cost
min_row = df1.loc[df1['best_cost'].idxmin()]
best_cost_value = min_row['best_cost']
best_thigh = min_row['thigh_length']
best_calf = min_row['calf_length']
best_ori_l = min_row['ori_l']
best_ori_theta = min_row['ori_theta']
best_hip_left_ratio = min_row['gear_left_ratio']
best_hip_right_ratio = min_row['gear_right_ratio']
best_hip_left_motor = min_row['motor_left_name']
best_hip_right_motor = min_row['motor_right_name']
best_gearbox_left = min_row['gearbox_left']
best_gearbox_right = min_row['gearbox_right']
best_ac1 = min_row['ac1']
best_ac2 = min_row['ac2']
best_ac3 = min_row['ac3']


# best_cost_value = min_row['best_cost']
# best_thigh_left = min_row['thigh_left_length']
# best_calf_left = min_row['calf_left_length']
# best_thigh_right = min_row['thigh_right_length']
# best_calf_right = min_row['calf_right_length']
# best_hip_left_ratio = min_row['gear_left_ratio']
# best_hip_right_ratio = min_row['gear_right_ratio']
# best_hip_left_motor = min_row['motor_left_name']
# best_hip_right_motor = min_row['motor_right_name']
# best_gearbox_left = min_row['gearbox_left']
# best_gearbox_right = min_row['gearbox_right']
# best_ac1 = min_row['ac1']
# best_ac2 = min_row['ac2']
# best_ac3 = min_row['ac3']

# min_row = df1.loc[df1['best_cost'].idxmin()]
# best_cost_value = min_row['best_cost']
# best_thigh = min_row['thigh_length']
# best_calf = min_row['calf_length']
# best_hip_ratio = min_row['gear_hip_ratio']
# best_knee_ratio = min_row['gear_knee_ratio']
# best_hip_left_motor = min_row['motor_hip_name']
# best_hip_right_motor = min_row['motor_knee_name']
# best_gearbox_left = min_row['gearbox_hip']
# best_gearbox_right = min_row['gearbox_knee']
# best_ac1 = min_row['ac1']
# best_ac2 = min_row['ac2']
# best_ac3 = min_row['ac3']

primary_data = min_row.to_dict()

# # # Load secondary CSV


# # # Ensure required columns exist in secondary CSV
# missing_secondary = [col for col in required_cols if col not in df2.columns]
# if missing_secondary:
#     raise ValueError(f"Secondary CSV missing columns: {missing_secondary}")

# # # if 'Cost' in df2.columns and 'Best Cost' not in df2.columns:
# # #     df2 = df2.rename(columns={'mass': 'Best Cost'})
# # # Match: First by Best Cost (float tolerance)

required_cols_secondary = list(df2.columns)
required_secondary_keys = [
    "Cost",
    "ac1",
    "ac2",
    "ac3",
    "Thigh",
    "Calf",
    "ori_l",
    "ori_theta",
]
missing_secondary = [col for col in required_secondary_keys if col not in df2.columns]
if missing_secondary:
    raise ValueError(f"Secondary CSV missing columns: {missing_secondary}")

# # for col in required_cols_secondary:
# #     df2[col] = pd.to_numeric(df2[col], errors='coerce')
tolerance = 1e-20
# #the filtered rows should have the same thigh length and calf length as the values with the thigh and calf length of the cost with the least value

#filtered_rows = df2[np.isclose(df2['Thigh_left_length'], best_thigh_left, atol=tolerance) & np.isclose(df2['Calf_left_length'], best_calf_left, atol=tolerance) & np.isclose(df2['Thigh_right_length'], best_thigh_right, atol=tolerance) & np.isclose(df2['Calf_right_length'], best_calf_right, atol=tolerance) & np.isclose(df2['Hip left ratio'], best_hip_left_ratio, atol=tolerance) & np.isclose(df2['Hip right ratio'], best_hip_right_ratio, atol=tolerance)  & np.isclose(df2['ac1'], best_ac1, atol=tolerance) & np.isclose(df2['ac2'], best_ac2, atol=tolerance) & np.isclose(df2['ac3'], best_ac3, atol=tolerance) ]
filtered_rows = df2[
    np.isclose(df2["ac1"], best_ac1, atol=tolerance)
    & np.isclose(df2["ac2"], best_ac2, atol=tolerance)
    & np.isclose(df2["ac3"], best_ac3, atol=tolerance)
    & np.isclose(df2["Cost"], best_cost_value, atol=tolerance)
    & np.isclose(df2["Thigh"], best_thigh, atol=tolerance)
    & np.isclose(df2["Calf"], best_calf, atol=tolerance)
    & np.isclose(df2["ori_l"], best_ori_l, atol=tolerance)
    & np.isclose(df2["ori_theta"], best_ori_theta, atol=tolerance)
]

# # # Match: Then by Thigh, Calf, Hip gear ratio, Knee gear ratio (also with tolerance)
# # for col in ['Thigh','Calf','Hip left motor','Hip right motor','Hip left ratio','Hip right ratio','Gearbox left','Gearbox right','Torso distance','Best X velocity','Average energy','Max height','Max distance','Unique id','ac1','ac2','ac3']:
# #     filtered_rows = filtered_rows[np.isclose(filtered_rows[col], min_row[col], atol=tolerance)]

print(f"Step 2: Found {len(filtered_rows)} rows after matching parameters.")
if not filtered_rows.empty:
    filtered_rows = filtered_rows.iloc[[0]]

# # # Convert matching rows to text
if filtered_rows.empty:
    secondary_data = None
    secondary_status = {
        "message": "No matching entries found in secondary CSV.",
        "thigh_length": float(best_thigh),
        "calf_length": float(best_calf),
        "best_cost": float(best_cost_value),
        "ori_l": float(best_ori_l),
        "ori_theta": float(best_ori_theta),
    }
else:
    secondary_data = filtered_rows.iloc[0].to_dict()
    secondary_status = None

final_data = {
    "primary": primary_data,
    "secondary": secondary_data,
    "secondary_status": secondary_status,
}

print(json.dumps(final_data, indent=2, default=str))
with open(output_json, 'w') as f:
    json.dump(final_data, f, indent=2, default=str)
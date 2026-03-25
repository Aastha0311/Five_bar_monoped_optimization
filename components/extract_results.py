import json
import numpy as np
import pandas as pd

primary_csv = '/home/stochlab/repo/optimal-design-legged-robots/results/CMAES_output/baseline/best_samples.csv'
secondary_csv = '/home/stochlab/repo/optimal-design-legged-robots/results/CMAES_output/baseline/all_samples.csv'
output_json = '/home/stochlab/repo/optimal-design-legged-robots/results/Opt_design_control_parameters/Nominal.json'

df1 = pd.read_csv(primary_csv)
df2 = pd.read_csv(secondary_csv)

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

min_row = df1.loc[df1['best_cost'].idxmin()]
best_cost_value = min_row['best_cost']
best_thigh = min_row['thigh_length']
best_calf = min_row['calf_length']
best_ori_l = min_row['ori_l']
best_ori_theta = min_row['ori_theta']
best_ac1 = min_row['ac1']
best_ac2 = min_row['ac2']
best_ac3 = min_row['ac3']




primary_data = min_row.to_dict()

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

tolerance = 1e-20
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
import pandas as pd
import numpy as np

# File paths
primary_csv = '/home/stochlab/repo/optimal-design-legged-robots/results/planar/dist/2bar/best_20_comb_030_070_2026-03-21_06-41-51_5.0.csv'
secondary_csv = '/home/stochlab/repo/optimal-design-legged-robots/results/planar/dist/2bar/all_20_comb_030_070_2026-03-21_06-41-51_5.0.csv'
output_txt = '/home/stochlab/repo/optimal-design-legged-robots/results/analysis/2bar_20.txt'

# Load primary CSV
df1 = pd.read_csv(primary_csv)
df2 = pd.read_csv(secondary_csv)
print(df1.columns)
print(df2.columns)

#Ensure required columns
# required_cols = ['thigh_length', 'calf_length', 'motor_left_name', 'motor_right_name',
#        'gear_left_ratio', 'gear_right_ratio', 'gearbox_left', 'gearbox_right',
#        'torso_distance', 'best_index', 'best_cost', 'ac1', 'ac2', 'ac3']
required_cols = ['thigh_length', 'calf_length', 'motor_hip_name', 'motor_knee_name',
       'gear_hip_ratio', 'gear_knee_ratio', 'gearbox_hip', 'gearbox_knee',
       'ik_height', 'best_index', 'best_cost', 'ac1', 'ac2', 'ac3']
missing_cols = [col for col in required_cols if col not in df1.columns]
if missing_cols:
    raise ValueError(f"Primary CSV missing columns: {missing_cols}")

for col in required_cols:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')

# # # Get row with lowest Best Cost
min_row = df1.loc[df1['best_cost'].idxmin()]
# best_cost_value = min_row['best_cost']
# best_thigh = min_row['thigh_length']
# best_calf = min_row['calf_length']
# best_hip_left_ratio = min_row['gear_left_ratio']
# best_hip_right_ratio = min_row['gear_right_ratio']
# best_hip_left_motor = min_row['motor_left_name']
# best_hip_right_motor = min_row['motor_right_name']
# best_gearbox_left = min_row['gearbox_left']
# best_gearbox_right = min_row['gearbox_right']
# best_ac1 = min_row['ac1']
# best_ac2 = min_row['ac2']
# best_ac3 = min_row['ac3']

min_row = df1.loc[df1['best_cost'].idxmin()]
best_cost_value = min_row['best_cost']
best_thigh = min_row['thigh_length']
best_calf = min_row['calf_length']
best_hip_ratio = min_row['gear_hip_ratio']
best_knee_ratio = min_row['gear_knee_ratio']
best_hip_left_motor = min_row['motor_hip_name']
best_hip_right_motor = min_row['motor_knee_name']
best_gearbox_left = min_row['gearbox_hip']
best_gearbox_right = min_row['gearbox_knee']
best_ac1 = min_row['ac1']
best_ac2 = min_row['ac2']
best_ac3 = min_row['ac3']

# # # Convert primary row to text
primary_output = ["Primary CSV - Entry with Lowest Best Cost:\n"]
primary_output += [f"{col}: {min_row[col]}" for col in df1.columns]
primary_text = "\n".join(primary_output)

# # # Load secondary CSV


# # # Ensure required columns exist in secondary CSV
# missing_secondary = [col for col in required_cols if col not in df2.columns]
# if missing_secondary:
#     raise ValueError(f"Secondary CSV missing columns: {missing_secondary}")

# # # if 'Cost' in df2.columns and 'Best Cost' not in df2.columns:
# # #     df2 = df2.rename(columns={'mass': 'Best Cost'})
# # # Match: First by Best Cost (float tolerance)

required_cols_secondary = ['Thigh', 'Calf', 'Hip left motor', 'Hip right motor', 'Hip left ratio',
       'Hip right ratio', 'Gearbox left', 'Gearbox right', 'Torso distance',
       'Best X velocity', 'Average energy', 'Max height', 'Max distance',
       'Unique id', 'Cost','ac1', 'ac2', 'ac3']

# required_cols_secondary= ['Thigh', 'Calf', 'Hip motor', 'Knee motor', 'Hip ratio', 'Knee ratio',
#        'Gearbox hip', 'Gearbox knee', 'Efficiency hip', 'Efficiency knee',
#        'Torso distance', 'ik_height', 'Best X velocity', 'Average energy',
#        'Max height', 'Max distance', 'Unique id', 'ac1', 'ac2', 'ac3']

missing_secondary = [col for col in required_cols_secondary if col not in df2.columns]
if missing_secondary:
    raise ValueError(f"Secondary CSV missing columns: {missing_secondary}")

# # for col in required_cols_secondary:
# #     df2[col] = pd.to_numeric(df2[col], errors='coerce')
tolerance = 1e-20
# #the filtered rows should have the same thigh length and calf length as the values with the thigh and calf length of the cost with the least value

filtered_rows = df2[np.isclose(df2['Thigh'], best_thigh, atol=tolerance) & np.isclose(df2['Calf'], best_calf, atol=tolerance) & np.isclose(df2['Hip left ratio'], best_hip_left_ratio, atol=tolerance) & np.isclose(df2['Hip right ratio'], best_hip_right_ratio, atol=tolerance)  & np.isclose(df2['ac1'], best_ac1, atol=tolerance) & np.isclose(df2['ac2'], best_ac2, atol=tolerance) & np.isclose(df2['ac3'], best_ac3, atol=tolerance) ]
#filtered_rows = df2[ np.isclose(df2['ac1'], best_ac1, atol=tolerance) & np.isclose(df2['ac2'], best_ac2, atol=tolerance) & np.isclose(df2['ac3'], best_ac3, atol=tolerance)]

# # # Match: Then by Thigh, Calf, Hip gear ratio, Knee gear ratio (also with tolerance)
# # for col in ['Thigh','Calf','Hip left motor','Hip right motor','Hip left ratio','Hip right ratio','Gearbox left','Gearbox right','Torso distance','Best X velocity','Average energy','Max height','Max distance','Unique id','ac1','ac2','ac3']:
# #     filtered_rows = filtered_rows[np.isclose(filtered_rows[col], min_row[col], atol=tolerance)]

print(f"Step 2: Found {len(filtered_rows)} rows after matching parameters.")
if not filtered_rows.empty:
    filtered_rows = filtered_rows.iloc[[0]]

# # # Convert matching rows to text
if filtered_rows.empty:
    secondary_text = (
        f"\n\nNo matching entries found in secondary CSV for:\n"
        f"Thigh Length: {best_thigh}\n"
        f"Calf Length: {best_calf}\n"
        f"Best Cost: {best_cost_value}\n"
        f"Parameters: {', '.join([f'{col}: {min_row[col]}' for col in df1.columns])}\n"
    )  
else:
    secondary_output = ["\n\nMatching Entry/Entries from Secondary CSV:\n"]
    for idx, row in filtered_rows.iterrows():
        secondary_output += [f"{col}: {row[col]}" for col in df2.columns]
        secondary_output.append("\n" + "-"*40 + "\n")
    secondary_text = "\n".join(secondary_output)

# # Final output
final_text = primary_text + secondary_text

# Print and save
print(final_text)
with open(output_txt, 'a') as f:
    f.write(final_text)
# import pandas as pd

# # File paths
# best_results_path = "/home/stochlab/repo/optimal-design-legged-robots/results/planar/comb/best_comb_065_035_2026-03-10_19-34-11_5.0.csv"
# all_results_path = "/home/stochlab/repo/optimal-design-legged-robots/results/planar/comb/all_comb_065_035_2026-03-10_19-34-11_5.0.csv"
# output_file = "/home/stochlab/repo/optimal-design-legged-robots/components/analysis/best_configuration_details.txt"

# # Load CSVs
# best_df = pd.read_csv(best_results_path)
# all_df = pd.read_csv(all_results_path)

# # ---------------------------------------------------
# # 1. Find row with minimum best_cost
# # ---------------------------------------------------
# min_idx = best_df["best_cost"].idxmin()
# best_row = best_df.loc[min_idx]

# # Extract actuator parameters
# thigh = best_row["thigh"]
# calf = best_row["calf"]
# hip_right = best_row["hip_right"]
# hip_left = best_row["hip_left"]

# # ---------------------------------------------------
# # 2. Find matching row in all_results
# # ---------------------------------------------------
# matched_rows = all_df[
#     (all_df["thigh"] == thigh) &
#     (all_df["calf"] == calf) &
#     (all_df["hip_right"] == hip_right) &
#     (all_df["hip_left"] == hip_left)
# ]

# # Extract v2_actual → v6_energy columns
# cols_to_extract = all_df.loc[:, "v2_actual":"v6_energy"]

# matched_metrics = matched_rows.loc[:, "v2_actual":"v6_energy"]

# # ---------------------------------------------------
# # 3. Write results to TXT
# # ---------------------------------------------------
# with open(output_file, "w") as f:

#     f.write("===== BEST RESULT (minimum best_cost) =====\n")
#     f.write(best_row.to_string())
#     f.write("\n\n")

#     f.write("===== MATCHING PERFORMANCE METRICS (from all_results) =====\n")
    
#     if not matched_metrics.empty:
#         f.write(matched_metrics.to_string(index=False))
#     else:
#         f.write("No matching row found in all_results.csv")

# print(f"Results saved to {output_file}")

import pandas as pd
import numpy as np

# -----------------------------
# File paths
# -----------------------------
best_results_path = "/home/stochlab/repo/optimal-design-legged-robots/results/planar/comb/best_comb_065_035_2026-03-10_19-34-11_5.0.csv"
all_results_path = "/home/stochlab/repo/optimal-design-legged-robots/results/planar/comb/all_comb_065_035_2026-03-10_19-34-11_5.0.csv"
output_file = "/home/stochlab/repo/optimal-design-legged-robots/components/analysis/best_configuration_details.txt"

# -----------------------------
# Load data
# -----------------------------
best_df = pd.read_csv(best_results_path)
all_df = pd.read_csv(all_results_path)

# -----------------------------
# 1. Find minimum cost row
# -----------------------------
min_idx = best_df[" best_cost"].idxmin()
best_row = best_df.loc[min_idx]

thigh = best_row["Thigh"]
calf = best_row["Calf"]
# hip_right = best_row[" Hip Right Motor"]
# hip_left = best_row["Hip Left Motor"]

# -----------------------------
# 2. Match actuator configuration in all_results
# -----------------------------
# tol = 1e-5

# mask = (
#     (np.abs(all_df["Thigh"] - thigh) < tol) &
#     (np.abs(all_df["Calf"] - calf) < tol) &
#     (np.abs(all_df["Hip_Right_Motor"] - hip_right) < tol) &
#     (np.abs(all_df["Hip_Left_Motor"] - hip_left) < tol)
# )

matched_rows = all_df[
    (all_df["Thigh"] == thigh) &
    (all_df["Calf"] == calf) 
    # (all_df[" Hip Right Motor"] == hip_right) &
    # (all_df["Hip Left Motor"] == hip_left)
]

#matched_rows = all_df[mask]

# Extract v2_actual → v6_energy
metrics = matched_rows.loc[:, "v2_actual":"v6_energy"]

# -----------------------------
# 3. Write output
# -----------------------------
with open(output_file, "w") as f:

    f.write("=================================================\n")
    f.write("BEST CMA-ES RESULT (Minimum best_cost)\n")
    f.write("=================================================\n\n")

    f.write(best_row.to_string())
    f.write("\n\n")

    f.write("-------------------------------------------------\n")
    f.write("Matched actuator configuration\n")
    f.write("-------------------------------------------------\n")

    f.write(f"thigh     : {thigh}\n")
    f.write(f"calf      : {calf}\n")
    # f.write(f"hip_right : {hip_right}\n")
    # f.write(f"hip_left  : {hip_left}\n\n")

    f.write("-------------------------------------------------\n")
    f.write("Performance Metrics (v2_actual → v6_energy)\n")
    f.write("-------------------------------------------------\n\n")

    if metrics.empty:
        f.write("No matching rows found in all_results.csv\n")
    else:
        for i, row in metrics.iterrows():
            f.write(f"Match {i}\n")
            for col, val in row.items():
                f.write(f"{col}: {val}\n")
            f.write("\n")

print(f"Saved results to: {output_file}")
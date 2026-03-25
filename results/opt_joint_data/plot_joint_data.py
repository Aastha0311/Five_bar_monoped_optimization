import json
import os
import pandas as pd
import matplotlib.pyplot as plt

CASE = "Nominal"  # Choose: A, B, C, or Nominal
SAVE_FIG = False
OUTPUT_DIR = "/home/stochlab/repo/optimal-design-legged-robots/results/opt_joint_data"

CASE_CHOICES = {
	"A": {
		"csv_path": "/home/stochlab/repo/optimal-design-legged-robots/results/opt_joint_data/Case_A_Jump_Timeseries.csv",
		"params_path": "/home/stochlab/repo/optimal-design-legged-robots/results/Opt_design_control_parameters/CaseA_ll.json",
	},
	"B": {
		"csv_path": "/home/stochlab/repo/optimal-design-legged-robots/results/opt_joint_data/Case_B_Jump_Timeseries.csv",
		"params_path": "/home/stochlab/repo/optimal-design-legged-robots/results/Opt_design_control_parameters/CaseB_gear_opt.json",
	},
	"C": {
		"csv_path": "/home/stochlab/repo/optimal-design-legged-robots/results/opt_joint_data/Case_C_Jump_Timeseries.csv",
		"params_path": "/home/stochlab/repo/optimal-design-legged-robots/results/Opt_design_control_parameters/CaseC_full_codesign_opt.json",
	},
	"NOMINAL": {
		"csv_path": "/home/stochlab/repo/optimal-design-legged-robots/results/opt_joint_data/Nominal_Jump_Timeseries.csv",
		"params_path": "/home/stochlab/repo/optimal-design-legged-robots/results/Opt_design_control_parameters/Nominal.json",
	},
}

case_key = CASE.strip().upper()
if case_key == "NOMINAL":
	case_key = "NOMINAL"
elif case_key not in CASE_CHOICES:
	raise ValueError("Invalid case. Choose A, B, C, or Nominal.")

case_label = "Nominal" if case_key == "NOMINAL" else f"Case_{case_key}"

csv_path = CASE_CHOICES[case_key]["csv_path"]
params_path = CASE_CHOICES[case_key]["params_path"]

df = pd.read_csv(csv_path)

with open(params_path, "r") as handle:
	params = json.load(handle)

ik_height = params["secondary"]["ik_height"]
df["slide_z"] = df["slide_z"] + ik_height
df["slide_x"] = df["slide_x"].abs()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

# 1) slide x/z
axes[0].plot(df["time"], df["slide_x"], label="x(m)")
axes[0].plot(df["time"], df["slide_z"], label="z(m)")
axes[0].set_ylabel("Jump distance/height[m]")
axes[0].legend()

# 2) hip/knee
axes[1].plot(df["time"], df["left_hip"], label="L hip")
axes[1].plot(df["time"], df["right_hip"], label="R hip")
axes[1].plot(df["time"], df["left_knee"], label="L knee")
axes[1].plot(df["time"], df["right_knee"], label="R knee")
axes[1].set_ylabel("Joint angles[rad]")
axes[1].legend()

# 3) ctrl
axes[2].plot(df["time"], df["ctrl_left"], label="L hip")
axes[2].plot(df["time"], df["ctrl_right"], label="R hip")
axes[2].set_ylabel("Joint Torques[Nm]")
for ax in axes:
	ax.set_xlabel("time (s)")
axes[2].legend()

plt.tight_layout()
if SAVE_FIG:
	output_path = os.path.join(OUTPUT_DIR, f"joint_data_{case_label}.png")
	plt.savefig(output_path, dpi=200)
plt.show()
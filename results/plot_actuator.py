# import pandas as pd
# import matplotlib.pyplot as plt

# # Load data
# df = pd.read_csv("/home/stochlab/repo/optimal-design-legged-robots/results/optimal_gearbox_selection.csv")

# # Sort data (important for clean lines)
# df = df.sort_values(by=["motor", "actual_ratio"])

# # Plot setup (paper-friendly)
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 12,
#     "figure.figsize": (12, 8),   # good for papers
#     "lines.linewidth":0.5,
#     "lines.markersize": 4
# })

# fig, ax = plt.subplots()

# # Plot one line per motor
# for motor, group in df.groupby("motor"):
#     ax.plot(
#         group["actual_ratio"],
#         group["mass"],
#         marker='o',
#         label=motor
#     )

# # Labels
# ax.set_xlabel("Gear Ratio")
# ax.set_ylabel("Mass (kg)")

# # Grid and legend
# ax.grid(True, linestyle='--', alpha=0.2)
# ax.legend(title="Motor", fontsize=9)

# # Tight layout for paper
# plt.tight_layout()

# # Save as vector (important for paper)
# plt.savefig("mass_vs_gear_ratio.pdf")
# plt.savefig("mass_vs_gear_ratio.png", dpi=300)

# plt.show()


## Subplots version (for detailed comparison) - not as paper-friendly but good for presentations or detailed analysis


# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("/home/stochlab/repo/optimal-design-legged-robots/results/optimal_gearbox_selection.csv")
# df = df.sort_values(by=["motor", "actual_ratio"])

# motors = df["motor"].unique()

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 10,
#     "figure.figsize": (12, 12),
# })

# fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
# axes = axes.flatten()

# for i, motor in enumerate(motors):
#     ax = axes[i]
#     group = df[df["motor"] == motor]

#     ax.plot(
#         group["actual_ratio"],
#         group["mass"],
#         marker='o',
#         linestyle='-'
#     )

#     ax.set_title(motor, fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.2)

# # Common labels
# fig.supxlabel("Gear Ratio")
# fig.supylabel("Mass (kg)")

# plt.tight_layout()
# plt.savefig("mass_vs_ratio_subplots.pdf")
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/home/stochlab/repo/optimal-design-legged-robots/results/optimal_gearbox_selection.csv")

# ---- SELECT MOTOR ----
motor_name = "MAD_M6C12"   # <-- change this to any motor you want

df_motor = df[df["motor"] == motor_name].copy()
df_motor = df_motor.sort_values(by="actual_ratio")

# ---- PLOT SETTINGS ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.figsize": (12, 9),
    "lines.linewidth": 1.5,
    "lines.markersize": 4
})

# ---- CREATE SUBPLOTS ----
fig, axes = plt.subplots(2, 1, sharex=True)

# ---- MASS vs GEAR RATIO ----
axes[0].plot(
    df_motor["actual_ratio"],
    df_motor["mass"],
    marker='o'
)
axes[0].set_ylabel("Mass (kg)")
axes[0].set_title(f"{motor_name}")
axes[0].grid(True, linestyle='--', alpha=0.5)

# ---- EFFICIENCY vs GEAR RATIO ----
axes[1].plot(
    df_motor["actual_ratio"],
    df_motor["efficiency"] * 100,  # convert to %
    marker='o'
)
axes[1].set_xlabel("Gear Ratio")
axes[1].set_ylabel("Efficiency (%)")
axes[1].grid(True, linestyle='--', alpha=0.5)

# ---- FINAL TOUCHES ----
plt.tight_layout()

# Save (vector format for paper)
plt.savefig(f"{motor_name}_mass_eff_vs_ratio.pdf")
plt.savefig(f"{motor_name}_mass_eff_vs_ratio.png", dpi=300)

plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load data
# df = pd.read_csv("/home/stochlab/repo/optimal-design-legged-robots/results/optimal_gearbox_selection.csv")

# # ---- SELECT MOTOR ----
# motor_name = "MAD_M6C12"   # <-- change this to any motor you want

# df_motor = df[df["motor"] == motor_name].copy()
# df_motor = df_motor.sort_values(by="actual_ratio")

# # ---- PLOT SETTINGS ----
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 11,
#     "figure.figsize": (12, 9),
#     "lines.linewidth": 1.5,
#     "lines.markersize": 4
# })

# # ---- CREATE SUBPLOTS ----
# fig, axes = plt.subplots(2, 1, sharex=True)

# # ---- MASS vs GEAR RATIO ----
# axes[0].plot(
#     df_motor["actual_ratio"],
#     df_motor["mass"],
#     marker='o'
# )
# axes[0].set_ylabel("Mass (kg)")
# axes[0].set_title(f"{motor_name}")
# axes[0].grid(True, linestyle='--', alpha=0.5)

# # ---- EFFICIENCY vs GEAR RATIO ----
# axes[1].plot(
#     df_motor["actual_ratio"],
#     df_motor["efficiency"] * 100,  # convert to %
#     marker='o'
# )
# axes[1].set_xlabel("Gear Ratio")
# axes[1].set_ylabel("Efficiency (%)")
# axes[1].grid(True, linestyle='--', alpha=0.5)

# # ---- FINAL TOUCHES ----
# plt.tight_layout()

# # Save (vector format for paper)
# plt.savefig(f"{motor_name}_mass_eff_type_vs_ratio.pdf")
# plt.savefig(f"{motor_name}_mass_eff_type_vs_ratio.png", dpi=300)

# plt.show()
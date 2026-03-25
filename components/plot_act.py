import os
import pandas as pd
import matplotlib.pyplot as plt

# CSV path relative to this script location
BASE_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
CSV_PATH = os.path.join(REPO_DIR, "results", "optimal_gearbox_selection.csv")

motor = input("Enter motor name (e.g., U8, U10, MN8014): ").strip()
df = pd.read_csv(CSV_PATH)

# Normalize column names just in case
df.columns = [c.strip() for c in df.columns]

required_cols = {"motor", "actual_ratio", "mass", "efficiency"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

df = df[df["actual_ratio"] <= 35].copy()
df.to_csv(CSV_PATH, index=False)

filtered = df[
    (df["motor"].str.strip().str.lower() == motor.lower())
].copy()

if filtered.empty:
    raise ValueError(f"No rows found for motor={motor} with actual_ratio <= 35.")

filtered = filtered.sort_values("actual_ratio")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

axes[0].plot(filtered["actual_ratio"], filtered["mass"], marker="o")
axes[0].set_title(f"{motor} Mass vs Actual Ratio")
axes[0].set_xlabel("Actual Ratio")
axes[0].set_ylabel("Mass (kg)")

axes[1].plot(filtered["actual_ratio"], filtered["efficiency"], marker="o")
axes[1].set_title(f"{motor} Efficiency vs Actual Ratio")
axes[1].set_xlabel("Actual Ratio")
axes[1].set_ylabel("Efficiency")

plt.tight_layout()
plots_dir = os.path.join(REPO_DIR, "results", "act_opt_plots")
os.makedirs(plots_dir, exist_ok=True)
output_path = os.path.join(plots_dir, f"gearbox_plots_{motor}.png")
plt.savefig(output_path, dpi=200)
plt.show()
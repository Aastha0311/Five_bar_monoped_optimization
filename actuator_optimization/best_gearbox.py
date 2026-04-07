import pandas as pd
import numpy as np
import os
import glob

BASE_DIR = os.path.dirname(__file__)
RESULT_ROOT = os.path.join(BASE_DIR, "results")

MOTORS = [
    "U8",
    "U10",
    "U12",
    "MN8014",
    "VT8020",
    "MAD_M6C12"
]

GR_MIN = 4
GR_MAX = 60
STEP = 0.1

target_ratios = np.round(np.arange(GR_MIN, GR_MAX + STEP, STEP), 1)

final_rows = []


def load_csvs_for_motor(motor):

    folder = os.path.join(RESULT_ROOT, f"results_BruteForce_{motor}")

    csv_files = glob.glob(os.path.join(folder, "*.csv"))

    data = []

    for f in csv_files:

        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()

        if "SSPG" in f:
            gearbox = "SSPG"
        elif "CPG" in f:
            gearbox = "CPG"
        elif "WPG" in f:
            gearbox = "WPG"
        elif "DSPG" in f:
            gearbox = "DSPG"
        else:
            continue

        df["gearbox"] = gearbox

        data.append(df)

    return data


for motor in MOTORS:

    print("Processing:", motor)

    csv_dfs = load_csvs_for_motor(motor)
    

    MAX_RATIO_ERROR = 0.1

    for target in target_ratios:

        candidates = []

        for df in csv_dfs:

            idx = (df["gearRatio"] - target).abs().idxmin()

            row = df.loc[idx]

            error = abs(row["gearRatio"] - target)

            # discard gearbox if ratio too far
            if error <= MAX_RATIO_ERROR:

                candidates.append(row.to_dict())

        # if no gearbox supports this ratio
        if len(candidates) == 0:
            continue

        candidates_df = pd.DataFrame(candidates)

        best = candidates_df.sort_values("Cost").iloc[0]

        final_rows.append({
            "motor": motor,
            "target_ratio": float(target),
            "actual_ratio": float(best["gearRatio"]),
            "gearbox": best["gearbox"],
            "mass": float(best["mass"]),
            "efficiency": float(best["eff"]),
            "cost": float(best["Cost"]),
            "ratio_error": abs(best["gearRatio"] - target)
        })


final_df = pd.DataFrame(final_rows)

final_df.to_csv("optimal_gearbox_selection.csv", index=False)

print("Saved: optimal_gearbox_selection.csv")
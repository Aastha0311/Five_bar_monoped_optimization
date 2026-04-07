import argparse
import csv
import os
from typing import Dict, List

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def _prompt_if_missing(value: str, label: str) -> str:
    if value:
        return value
    return input(f"Enter {label}: ").strip()


def _collect_csv_files(results_dir: str) -> List[str]:
    csv_files: List[str] = []
    for root, _, files in os.walk(results_dir):
        for name in files:
            if name.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, name))
    return csv_files


def _matches_file(motor: str, gearbox: str, file_path: str) -> bool:
    name = os.path.basename(file_path).lower()
    return motor.lower() in name and gearbox.lower() in name


def _read_matches(file_path: str, ratio: float, tol: float) -> List[Dict[str, str]]:
    matches: List[Dict[str, str]] = []
    with open(file_path, "r", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if not reader.fieldnames:
            return matches
        for row in reader:
            try:
                row_ratio = float(row.get("gearRatio", ""))
            except (TypeError, ValueError):
                continue
            if abs(row_ratio - ratio) <= tol:
                matches.append(row)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find gearbox parameters for a motor, ratio, and gearbox type."
    )
    parser.add_argument("--motor", help="Motor name, e.g., U8")
    parser.add_argument("--ratio", type=float, help="Gear ratio to match")
    parser.add_argument("--gearbox", help="Gearbox type, e.g., WPG/SSPG/CPG")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for gear ratio match",
    )
    args = parser.parse_args()

    motor = _prompt_if_missing(args.motor, "motor name")
    gearbox = _prompt_if_missing(args.gearbox, "gearbox type")
    ratio_value = args.ratio
    if ratio_value is None:
        ratio_value = float(_prompt_if_missing("", "gear ratio"))

    csv_files = _collect_csv_files(RESULTS_DIR)
    if not csv_files:
        print(f"No CSV files found under {RESULTS_DIR}")
        return

    first_match = None
    for file_path in csv_files:
        if not _matches_file(motor, gearbox, file_path):
            continue
        rows = _read_matches(file_path, ratio_value, args.tolerance)
        if rows:
            first_match = rows[0]
            first_match["source_file"] = file_path
            break

    if not first_match:
        print("No matches found.")
        return

    fieldnames = list(first_match.keys())
    writer = csv.DictWriter(
        os.sys.stdout, fieldnames=fieldnames, extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerow(first_match)


if __name__ == "__main__":
    main()

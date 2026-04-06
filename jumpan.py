import csv
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}. "
            f"Put the file under: {DATA_DIR}"
        )

# Input and output file paths
input_file = DATA_DIR / "Jumps.csv"
output_file = OUTPUT_DIR / "processedjumps.csv"
_require_file(input_file)

# Dictionary to store total height and count for each date
date_data = defaultdict(lambda: {"total_height": 0, "count": 0})

# Read the input CSV and aggregate data
with input_file.open(mode="r", newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=";")
    for row in reader:
        date = row["Date"]
        try:
            height = int(row["HeightInCm"])
        except (ValueError, KeyError):
            # Skip rows with invalid or missing HeightInCm
            continue
        date_data[date]["total_height"] += height
        date_data[date]["count"] += 1

# Calculate averages and write to output CSV
with output_file.open(mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile, delimiter=";")
    writer.writerow(["Date", "AverageHeightInCm"])
    for date, data in date_data.items():
        average_height = data["total_height"] / data["count"]
        writer.writerow([date, round(average_height, 2)])

print(f"✅ Wrote: {output_file}")

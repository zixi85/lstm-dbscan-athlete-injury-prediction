import pandas as pd
import os
import matplotlib

SHOW_PLOTS = os.getenv("SHOW_PLOTS", "1") != "0"
SAVE_PLOTS = os.getenv("SAVE_PLOTS", "1") != "0"
if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _slug(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )


def _save_or_show(filename: str) -> None:
    if SAVE_PLOTS:
        out_path = FIGURES_DIR / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}. "
            f"Put the file under: {DATA_DIR}"
        )

# ----------------------------
# Load & clean the wellness data
# ----------------------------
file_path = DATA_DIR / "Wellness.csv"
_require_file(file_path)
wellness_data = pd.read_csv(file_path, sep=';')

# Convert 'Date' to datetime format
wellness_data['Date'] = pd.to_datetime(wellness_data['Date'], format='%d-%m-%Y', errors='coerce')

# Fill missing numeric values with column means
numeric_cols = wellness_data.select_dtypes(include=['float64', 'int64']).columns
wellness_data[numeric_cols] = wellness_data[numeric_cols].fillna(wellness_data[numeric_cols].mean())

# ----------------------------
# Analyze wellness distributions
# ----------------------------
def plot_distributions(data, columns):
    for col in columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[col], kde=True, bins=20)
        plt.title(f"Distribution of {col}")
        plt.grid(True)
        _save_or_show(f"wellness_distribution_{_slug(col)}.png")

wellness_metrics = ['Wellness', 'Mood', 'Recovered', 'Muscle Soreness', 'Sleep quality', 'Hours of sleep']
plot_distributions(wellness_data, wellness_metrics)

# ----------------------------
# Injury analysis from four key questions
# ----------------------------
injury_cols = ['Difficultparticipating', 'Reducedtraining', 'Affectedperformance', 'Symptomscomplaints']

# Ensure all injury-related columns are numeric (in case of read errors)
wellness_data[injury_cols] = wellness_data[injury_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# Define binary injury: 1 if any of the four questions ≥ 2
wellness_data['BinaryInjury'] = (
    (wellness_data[injury_cols] >= 2).any(axis=1)
).astype(int)

# (Optional) InjuryScore & InjuryCategory for additional insights
wellness_data['InjuryScore'] = wellness_data[injury_cols].sum(axis=1)

def categorize_injury(score):
    if score == 0:
        return 'No Injury'
    elif score <= 4:
        return 'Mild'
    elif score <= 8:
        return 'Moderate'
    else:
        return 'Severe'

wellness_data['InjuryCategory'] = wellness_data['InjuryScore'].apply(categorize_injury)

# ----------------------------
# Correlation heatmap
# ----------------------------
correlation_data = wellness_data[wellness_metrics + injury_cols]
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Between Wellness Metrics and Injury Questions")
_save_or_show("wellness_correlation_heatmap.png")

# ----------------------------
# Save the enhanced dataset
# ----------------------------
output_path = OUTPUT_DIR / "Wellness_Analyzed.csv"
wellness_data.to_csv(output_path, index=False)
print(f"✅ Updated data saved to {output_path}")

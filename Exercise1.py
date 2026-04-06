import pandas as pd
from itertools import combinations
from collections import Counter
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

# ========== Data Loading ==========
exercise_path = DATA_DIR / "ExerciseTrainingData.csv"
_require_file(exercise_path)
exercise_df = pd.read_csv(exercise_path, delimiter=';')

# Date parsing
if 'Date' in exercise_df.columns:
    exercise_df['Date'] = pd.to_datetime(exercise_df['Date'], dayfirst=True, errors='coerce')
else:
    exercise_df['Date'] = pd.to_datetime(exercise_df['TrainingID'].str[:10], dayfirst=True, errors='coerce')

# Drop rows with missing required values
exercise_df.dropna(subset=['TrainingID', 'TrainingSubtype', 'Duration_m'], inplace=True)

# Add `Duration_s` column (if missing)
if 'Duration_s' not in exercise_df.columns:
    exercise_df['Duration_s'] = exercise_df['Duration_m'] * 60

# ========== Daily TrainingType Combinations ==========
# Get unique daily training type combinations; deduplicate and sort for consistency (for comparison)
trainingtype_combinations = (
    exercise_df.groupby('Date')['TrainingType']
    .apply(lambda x: sorted(set(x)))  # Sort alphabetically to keep combination order consistent
    .reset_index()
    .rename(columns={'TrainingType': 'TrainingType_Combination'})
)

# Convert to string combinations for frequency analysis
trainingtype_combinations['CombinationStr'] = trainingtype_combinations['TrainingType_Combination'].apply(lambda x: ', '.join(x))

# Count the most frequent training type combinations
trainingtype_comb_freq = (
    trainingtype_combinations['CombinationStr']
    .value_counts()
    .reset_index()
    .rename(columns={'index': 'TrainingType_Combination', 'CombinationStr': 'Frequency'})
)

# ========== Daily Total Duration by TrainingType ==========
daily_type_duration = (
    exercise_df.groupby(['Date', 'TrainingType'])['Duration_s']
    .sum()
    .reset_index()
    .pivot(index='Date', columns='TrainingType', values='Duration_s')
    .fillna(0)
)

# ========== Daily TrainingType Ratios (rounded to 2 decimals) ==========
daily_type_ratio = daily_type_duration.div(daily_type_duration.sum(axis=1), axis=0)
daily_type_ratio = daily_type_ratio.round(2)

# ========== Save Results ==========
trainingtype_combinations.to_csv(OUTPUT_DIR / "daily_training_combinations.csv", index=False)
daily_type_duration.reset_index().to_csv(OUTPUT_DIR / "trainingtype_duration.csv", index=False)
daily_type_duration.reset_index().to_csv(OUTPUT_DIR / "daily_trainingtype_duration.csv", index=False)
daily_type_ratio.reset_index().to_csv(OUTPUT_DIR / "daily_trainingtype_ratio.csv", index=False)
trainingtype_comb_freq.to_csv(OUTPUT_DIR / "frequent_training_combinations.csv", index=False)

print(
    "✅ Wrote: daily_training_combinations.csv, trainingtype_duration.csv, "
    "daily_trainingtype_duration.csv, daily_trainingtype_ratio.csv, frequent_training_combinations.csv "
    f"to {OUTPUT_DIR}"
)

import pandas as pd
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

# ------------------------
# Step 1: Load and preprocess data sources
# ------------------------

def load_training_load():
    path = DATA_DIR / "training_load_trend.csv"
    _require_file(path)
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

def load_strength_training():
    path = OUTPUT_DIR / "ProcessedStrengthTraining.csv"
    _require_file(path)
    df = pd.read_csv(path, sep=";", parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    daily_volume = (
        df.groupby(["Date", "Exercise"])["Volume"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    return daily_volume

def load_injury_labels():
    path = OUTPUT_DIR / "Wellness_Analyzed.csv"
    _require_file(path)
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df[["Date", "BinaryInjury"]].rename(columns={"BinaryInjury": "Injury"})

def load_jump_features():
    path = OUTPUT_DIR / "processedjumps.csv"
    _require_file(path)
    df = pd.read_csv(path, sep=";", parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df.groupby("Date").mean(numeric_only=True).reset_index()

def load_training_type_duration():
    try:
        path = OUTPUT_DIR / "trainingtype_duration.csv"
        df = pd.read_csv(path, parse_dates=["Date"])
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        return df
    except FileNotFoundError:
        print("⚠️ trainingtype_duration.csv not found. Skipping training type merge.")
        return None

# ------------------------
# Step 2: Merge all data sources
# ------------------------

def merge_data():
    training_load = load_training_load()
    strength_daily = load_strength_training()
    injury_labels = load_injury_labels()
    jumps = load_jump_features()

    # Merge training load + strength
    merged = pd.merge(training_load, strength_daily, on="Date", how="outer")

    # Merge injury labels
    merged = pd.merge(merged, injury_labels, on="Date", how="left")

    # Merge jump metrics
    merged = pd.merge(merged, jumps, on="Date", how="left")

    # Sort columns (Injury at the end)
    cols = [c for c in merged.columns if c != "Injury"] + ["Injury"]
    merged = merged[cols].sort_values("Date").reset_index(drop=True)

    return merged

# ------------------------
# Step 3: Optionally merge training type data
# ------------------------

def merge_training_type(merged_df):
    training_type_df = load_training_type_duration()
    if training_type_df is not None:
        merged_with_type = pd.merge(merged_df, training_type_df, on="Date", how="left")
        merged_with_type = merged_with_type.sort_values("Date").reset_index(drop=True)
        out_path = OUTPUT_DIR / "FinalMergedData_WithTrainingType.csv"
        merged_with_type.to_csv(out_path, index=False, sep=";")
        print(f"✅ Saved: {out_path}")

# ------------------------
# Step 4: Save and report
# ------------------------

def save_and_report(merged_df):
    out_path = OUTPUT_DIR / "FinalMergedData.csv"
    merged_df.to_csv(out_path, index=False, sep=";")
    print(f"✅ Saved: {out_path}")

    missing_pct = merged_df.isnull().mean().round(3) * 100
    missing_report = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if not missing_report.empty:
        print("\n📋 Missing value percentage:")
        print(missing_report)
    else:
        print("\n🎉 No missing values!")

# ------------------------
# Main Execution
# ------------------------

if __name__ == "__main__":
    print("🚀 Starting data merge...")

    merged_df = merge_data()
    merge_training_type(merged_df)
    save_and_report(merged_df)

    print("✅ All done.")

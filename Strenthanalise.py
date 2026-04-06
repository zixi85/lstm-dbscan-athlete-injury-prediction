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

# Load the CSV file
file_path = DATA_DIR / "StrengthTraining.csv"
_require_file(file_path)
data = pd.read_csv(file_path, sep=';')

# Convert columns to appropriate data types
data['Weight'] = data['Weight'].str.replace(',', '.').astype(float, errors='ignore')
data['Prct'] = data['Prct'].str.replace(',', '.').astype(float, errors='ignore')

# Convert the Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Group by Date and Exercise, aggregating numeric columns
aggregated_data = data.groupby(['Date', 'Exercise'], as_index=False).agg({
    'Reps': 'sum',
    'Prct': 'mean',
    'Weight': 'mean'
})

# Fill missing values with the mean of their respective Exercise type
for column in ['Reps', 'Prct', 'Weight']:
    aggregated_data[column] = aggregated_data.groupby('Exercise')[column].transform(
        lambda x: x.fillna(x.mean())
    )

# Calculate Volume for each row
aggregated_data['Volume'] = aggregated_data['Reps'] * aggregated_data['Weight']

# Calculate TotalVolume for each day
aggregated_data['TotalVolume'] = aggregated_data.groupby('Date')['Volume'].transform('sum')

# Round all numeric columns to one decimal place
numeric_columns = ['Reps', 'Prct', 'Weight', 'Volume', 'TotalVolume']
aggregated_data[numeric_columns] = aggregated_data[numeric_columns].round(1)

# Sort aggregated_data by Date in ascending order
aggregated_data = aggregated_data.sort_values(by='Date', ascending=True)

# Save the processed data to a new CSV file
output_path = OUTPUT_DIR / "ProcessedStrengthTraining.csv"
aggregated_data.to_csv(output_path, sep=';', index=False)

print(f"Processed data saved to {output_path}")

# Identify training combinations
# Create a pivot table to track the presence of each Exercise per Date
combination_matrix = data.pivot_table(index='Date', columns='Exercise', aggfunc='size', fill_value=0)
combination_matrix = (combination_matrix > 0).astype(int)  # Convert to boolean presence (1 for present, 0 for absent)

# Count the frequency of each combination
combination_matrix['Combination'] = combination_matrix.apply(lambda row: tuple(row), axis=1)
combination_frequencies = combination_matrix['Combination'].value_counts()

# Analyze differences in training volume
# Add a column to aggregated_data to represent the combination for each Date
aggregated_data['Combination'] = aggregated_data['Date'].map(combination_matrix['Combination'])

# Group by training combination and calculate average TotalVolume
combination_volume = aggregated_data.groupby('Combination')['TotalVolume'].mean().sort_values()

# Visualization: Heatmap of TotalVolume by Date and Exercise
heatmap_data = aggregated_data.pivot(index='Date', columns='Exercise', values='TotalVolume')

# Convert the Date index to string format for proper display
heatmap_data.index = heatmap_data.index.strftime('%d-%m-%Y')

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Heatmap of Total Volume by Date and Exercise")
_save_or_show("strength_heatmap_total_volume_by_date_exercise.png")

# Visualization: Boxplot of Volume by Exercise
plt.figure(figsize=(10, 6))
sns.boxplot(data=aggregated_data, x='Exercise', y='Volume')
plt.title("Boxplot of Volume by Exercise")
plt.xticks(rotation=45)
_save_or_show("strength_boxplot_volume_by_exercise.png")

# Visualization: Bar chart of TotalVolume by Exercise
total_volume_by_exercise = aggregated_data.groupby('Exercise')['TotalVolume'].sum().sort_values()
plt.figure(figsize=(10, 6))
total_volume_by_exercise.plot(kind='bar', color='skyblue')
plt.title("Total Volume by Exercise")
plt.ylabel("Total Volume")
plt.xticks(rotation=45)
_save_or_show("strength_barchart_total_volume_by_exercise.png")

# Visualization: Frequency of training combinations
plt.figure(figsize=(10, 6))
combination_frequencies.plot(kind='bar', color='skyblue')
plt.title("Frequency of Training Combinations")
plt.ylabel("Frequency")
plt.xlabel("Training Combination")
plt.xticks(rotation=45)
_save_or_show("strength_barchart_training_combination_frequency.png")

# Visualization: Average TotalVolume by training combination
plt.figure(figsize=(10, 6))
combination_volume.plot(kind='bar', color='orange')
plt.title("Average Total Volume by Training Combination")
plt.ylabel("Average Total Volume")
plt.xlabel("Training Combination")
plt.xticks(rotation=45)
_save_or_show("strength_barchart_avg_total_volume_by_combination.png")

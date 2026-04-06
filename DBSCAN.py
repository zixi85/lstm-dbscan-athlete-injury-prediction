import pandas as pd
import numpy as np
import os
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

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
            f"Run `mergedata.py` first, or put inputs under: {DATA_DIR}"
        )

# -----------------------------
# Step 1: Data Loading and Cleaning
# -----------------------------
input_path = OUTPUT_DIR / "FinalMergedData_WithTrainingType.csv"
_require_file(input_path)
df = pd.read_csv(input_path, sep=";")

# Clean column names (remove leading and trailing spaces)
df.columns = df.columns.str.strip()

# Fill critical missing values
df["Injury"] = df["Injury"].fillna(0)
df["AverageHeightInCm"] = df["AverageHeightInCm"].ffill()

# Some training features may be extremely sparse, set them to 0 or consider removing them
for col in ["Fullbody", "Lower", "Upper"]:
    df[col] = df[col].fillna(0)

# Fill all other missing values with 0
df.fillna(0, inplace=True)

# -----------------------------
# Step 2: Feature Selection and Standardization
# -----------------------------
features = df[[
    "Total_Duration_s", "Fullbody", "Lower", "Upper", "AverageHeightInCm",
    "Competition", "Complex 1", "Complex 2", "Complex total",
    "Match", "Physical", "Technique"
]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# -----------------------------
# Step 3: DBSCAN Clustering
# -----------------------------
db = DBSCAN(eps=1.5, min_samples=3)
df["Cluster"] = db.fit_predict(X_scaled)

injury_ratio = df.groupby("Cluster")["Injury"].mean().reset_index().rename(columns={"Injury": "Injury_Rate"})


sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.countplot(x="Cluster", hue="Injury", data=df)
plt.title("DBSCAN Cluster vs Injury Distribution")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.legend(title="Injury")
plt.tight_layout()
_save_or_show("dbscan_cluster_vs_injury_distribution.png")

injury_ratio.sort_values(by="Injury_Rate", ascending=False)
# -----------------------------
# Step 4: Cluster Analysis (Injury Rate)
# -----------------------------
injury_ratio = df.groupby("Cluster")["Injury"].mean().reset_index().rename(columns={"Injury": "Injury_Rate"})
print("📊 Injury rate for each cluster:")
print(injury_ratio.sort_values(by="Injury_Rate", ascending=False))

# Analyze feature distribution by cluster
cluster_features = df.groupby("Cluster")[features.columns].mean()
# Print the feature distribution for each cluster
cluster_features.to_csv(OUTPUT_DIR / "DBSCAN_Cluster_Features.csv", index=True)

# -----------------------------
# Step 5: PCA Visualization
# -----------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df["PCA1"], df["PCA2"] = pca_result[:, 0], pca_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="PCA1", y="PCA2",
    hue="Cluster",
    style="Injury",
    palette="tab10",
    s=100
)
plt.title("PCA Projection of Training Data with DBSCAN Clusters and Injury")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()
_save_or_show("dbscan_pca_projection_clusters_injury.png")

# -----------------------------
# Step 6: Optional - Save Cluster Results
# -----------------------------
# df.to_csv("DBSCAN_Clustered_TrainingData.csv", index=False)

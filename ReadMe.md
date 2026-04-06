Sports injuries are influenced by complex interactions between training load, recovery, physical readiness, and wellness. This project shows how machine learning can help uncover early warning signs from athlete monitoring data and support more proactive training management.
Rather than only predicting whether an injury may occur, this work also explores **why certain risk patterns emerge**, making the analysis more useful for coaches and sports scientists.
Using daily wellness questionnaires, training logs, strength records, and jump performance data from a Dutch national volleyball player, I built:

## ✨ Project Highlights
- an **LSTM-based time-series model** for injury prediction from multivariate daily athlete data
- a **DBSCAN clustering pipeline** to identify high-risk training patterns
- Identified a **high-risk cluster with 85.7% injury rate**
- Applied **Random Oversampling** and **class-weighted loss** to handle class imbalance

The project demonstrates an end-to-end sports data science workflow, from **data cleaning and feature engineering** to **predictive modeling, clustering, and interpretation**.

## 🔒 Data Availability

The original athlete monitoring data used in this project cannot be publicly released for confidentiality reasons.

Therefore:
- the raw dataset is **not provided** in this repository
- any paths, scripts, or file references are included for **demonstration purposes**
- this repository is intended to showcase the **analysis pipeline**, **modeling approach**, and **technical implementation** for similar sports analytics use cases
If needed, this workflow can be adapted to other longitudinal athlete monitoring datasets with comparable structure.

File Overview
===============

1. **mergedata.py**
   - Merges multiple data sources into a single day-level dataset.
   - Sources: strength training, jump tests, wellness (injury label), training loads, and training type durations.

2. **LSTM3.py**
   - Builds an LSTM model to predict future injury risk from historical daily data.
   - Includes oversampling and class-weight balancing to address label imbalance.
   - Visualizes training/validation loss and prints performance metrics.

3. **DBSCAN.py**
   - Performs DBSCAN clustering on daily training sessions.
   - Visualizes cluster-wise injury distribution and PCA projections.

4. **Exercise1.py**
   - Analyzes daily `TrainingType` usage:
     - Most frequent combinations
     - Duration per type
     - Proportion of training time per type

5. **Strenthanalise.py**
   - Processes strength training logs:
     - Computes total volume per day
     - Analyzes exercise combinations
   - Produces visualizations:
     - Heatmaps, boxplots, and bar charts of training volumes
     - Frequencies of training combination types

6. **jumpan.py**
   - Computes average jump height per day from raw jump data.

7. **wellness.py**
   - Cleans and analyzes wellness questionnaire data.
   - Computes binary injury labels and injury severity scores.
   - Correlates wellness indicators with injury responses.

---
How to Run
=============
1. Preprocess inputs (writes to `data/output/`):
   - `python wellness.py`
   - `python Strenthanalise.py`
   - `python jumpan.py`
   - `python Exercise1.py`

2. Merge day-level dataset:
   - `python mergedata.py`

3. Run modeling/analysis:
   - `python DBSCAN.py`
   - `python LSTM3.py`

> **Note**
This project was developed for academic purposes in a Sports Data Science.  
To protect confidential athlete information, the original dataset is not publicly available.  
The repository is shared as a methodological example of how similar injury prediction and training pattern analysis can be implemented in practice.
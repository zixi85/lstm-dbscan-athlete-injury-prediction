import pandas as pd
import numpy as np
import os
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

SHOW_PLOTS = os.getenv("SHOW_PLOTS", "1") != "0"
SAVE_PLOTS = os.getenv("SAVE_PLOTS", "1") != "0"
if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from imblearn.over_sampling import RandomOverSampler
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
            "Run `mergedata.py` first to generate it."
        )

# ------------------------------
def load_and_prepare_data(file_path):
    file_path = Path(file_path)
    _require_file(file_path)
    df = pd.read_csv(file_path, sep=";", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.drop(columns=["Date"])

    for col in df.columns:
        if "Match" in col or "Upper" in col or "Lower" in col or "Complex" in col or "Technique" in col or "Physical" in col:
            df[col] = df[col].fillna(0)
        elif "Height" in col or "Jump" in col:
            df[col] = df[col].ffill()
        else:
            df[col] = df[col].fillna(0)

    X = df.drop(columns=["Injury"])
    y = df["Injury"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values

# ------------------------------
def create_rolling_lstm_data(X, y, window=7, test_ratio=0.2):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    test_size = int(len(X_seq) * test_ratio)
    return (
        X_seq[:-test_size], X_seq[-test_size:],
        y_seq[:-test_size], y_seq[-test_size:]
    )

# ------------------------------
def oversample_data(X_train, y_train):
    X_flat = X_train.reshape(X_train.shape[0], -1)
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_flat, y_train)
    X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
    return X_resampled, y_resampled

# ------------------------------
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ------------------------------
def train_and_evaluate_lstm_with_balancing(file_path, window=7, epochs=50, batch_size=16):
    X_scaled, y = load_and_prepare_data(file_path)
    X_train, X_test, y_train, y_test = create_rolling_lstm_data(X_scaled, y, window=window)

    # Ensure y_train is integer for np.bincount
    y_train = y_train.astype(int)
    print(f"\n📊 原始训练集标签分布: {np.bincount(y_train)}")

    # === 标签过采样 ===
    X_bal, y_bal = oversample_data(X_train, y_train)
    y_bal = y_bal.astype(int)  # Ensure y_bal is integer for np.bincount
    print(f"✅ 过采样后标签分布: {np.bincount(y_bal)}")

    # === 计算 class_weight ===
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_bal), y=y_bal)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"⚖ class_weight: {class_weight_dict}")

    # === 模型训练 ===
    model = build_lstm_model(input_shape=(X_bal.shape[1], X_bal.shape[2]))
    history = model.fit(
        X_bal, y_bal,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        verbose=int(os.getenv("VERBOSE", "1"))
    )

    # === 评估 ===
    y_pred = model.predict(X_test).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_binary))

    # 可视化 loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.grid()
    _save_or_show("lstm_loss_curve.png")

    print("✅ LSTM run completed")
    return model, history


if __name__ == "__main__":
    model, history = train_and_evaluate_lstm_with_balancing(
        OUTPUT_DIR / "FinalMergedData_WithTrainingType.csv",
        window=int(os.getenv("WINDOW", "7")),
        epochs=int(os.getenv("EPOCHS", "50")),
        batch_size=int(os.getenv("BATCH_SIZE", "16")),
    )
import pandas as pd
import numpy as np
from math import floor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import joblib

# ---------- 1. LOAD DATA ----------
df = pd.read_csv("dataset.csv")   # make sure this file is in the repo root

# We assume df has these columns; rename here if needed
# df.columns -> check once in a notebook or print it

# ---------- 2. CREATE LAG FEATURES ----------
n_lag = 8
for i in range(3, n_lag + 1):
    df[f"count_lag_{i}"] = df["homeless_count"].shift(i)

df["median_rent_city_lag1"]      = df["ZORI"].shift(1)
df["avg_temp_lag1"]              = df["average_temp"].shift(1)
df["unemployment_rate_lag1"]     = df["unemployment_rate"].shift(1)
df["evictions_lag1"]             = df["evictions"].shift(1)
df["precipitation_lag1"]         = df["precipitation"].shift(1)
df["cpi_lag1"]                   = df["cpi"].shift(1)
df["cpi_lag2"]                   = df["cpi"].shift(2)
df["cpi_lag3"]                   = df["cpi"].shift(3)

# Drop features you said you don’t want
drop_cols = ["shelter_beds", "industrial_production"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ---------- 3. CLEAN MISSING VALUES FROM SHIFTS ----------
for col in df.columns:
    df[col] = df[col].infer_objects(copy=False)
    df[col] = df[col].interpolate()

# Drop the first 12 rows which have incomplete lags
df = df.iloc[12:].copy()
# Optionally drop last n_lag rows like in your script
df = df.iloc[:-n_lag].copy()

# Add explicit year / month, keep year_month for baseline later
df["dt"] = pd.to_datetime(df["year_month"])
df["month"] = df["dt"].dt.month
df["year"]  = df["dt"].dt.year

# ---------- 4. BUILD FEATURES / LABELS ----------
exclude_cols = {"homeless_count", "year_month", "dt"}
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
y = df["homeless_count"].values

# Time-series style split: last 20% as test, no shuffle
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

# ---------- 5. SCALE FEATURES ----------
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ---------- 6. DEFINE AND TRAIN KERAS MODEL ----------
model = Sequential([
    Dense(20, input_shape=(X_train_sc.shape[1],)),
    BatchNormalization(),
    Dense(25, activation="relu"),
    BatchNormalization(),
    Dense(20, activation="relu"),
    BatchNormalization(),
    Dense(20, activation="relu"),
    BatchNormalization(),
    Dense(20, activation="relu"),
    BatchNormalization(),
    Dense(10, activation="relu"),
    BatchNormalization(),
    Dense(1),
])

schedule = ExponentialDecay(0.05, decay_steps=2000, decay_rate=0.96)
opt = Adam(learning_rate=schedule)

model.compile(optimizer=opt, loss="mse")

model.fit(X_train_sc, y_train, epochs=500, batch_size=5, verbose=0)

# Evaluate once
y_pred_test = model.predict(X_test_sc).ravel()
print("Test R^2:", r2_score(y_test, y_pred_test))

# ---------- 7. SAVE ARTIFACTS ----------
# a) Keras model (saved in both newer .keras format for the app and legacy .h5 for debugging)
model.save("model_keras.keras")
model.save("model_keras.h5")

# b) Scaler and feature order
joblib.dump(scaler, "scaler.joblib")
joblib.dump(feature_cols, "feature_cols.joblib")

# c) Baseline row for last available month (for Jan 2024 scenario)
df_sorted = df.sort_values("dt")
last_row = df_sorted.iloc[[-1]][feature_cols + ["year_month"]]  # keep features + date
last_row.to_csv("last_row_2023.csv", index=False)
print("Artifacts saved: model_keras.keras, model_keras.h5, scaler.joblib, feature_cols.joblib, last_row_2023.csv")

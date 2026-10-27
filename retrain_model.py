from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wen_hea_hardness_155.csv"
MODEL_OUT = ROOT / "models" / "hea_hardness_model_train140.joblib"

ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Ni"]
# These are the 15 classroom candidates A–O in data/classroom_candidates.csv.
HOLDOUT_IDS = [154, 146, 141, 113, 111, 107, 100, 96, 85, 76, 59, 37, 14, 11, 5]

df = pd.read_csv(DATA)
X = df[ELEMENTS].div(df[ELEMENTS].sum(axis=1), axis=0).to_numpy() * 100.0
y = df["HV"].to_numpy()
mask_test = df["ID"].isin(HOLDOUT_IDS).to_numpy()
mask_train = ~mask_test

model = ExtraTreesRegressor(
    n_estimators=300,
    max_features=1.0,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
model.fit(X[mask_train], y[mask_train])
pred = model.predict(X[mask_test])

print(f"Train n: {mask_train.sum()} | Holdout n: {mask_test.sum()}")
print(f"MAE:  {mean_absolute_error(y[mask_test], pred):.3f} HV")
print(f"RMSE: {mean_squared_error(y[mask_test], pred) ** 0.5:.3f} HV")
print(f"R2:   {r2_score(y[mask_test], pred):.6f}")

joblib.dump(model, MODEL_OUT)
print(f"Saved model to: {MODEL_OUT}")

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "data" / "classroom_candidates.csv"
MODEL = ROOT / "models" / "hea_hardness_model_train140.joblib"
ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Ni"]

df = pd.read_csv(CAND).sort_values("Candidate").reset_index(drop=True)
model = joblib.load(MODEL)
X = df[ELEMENTS].div(df[ELEMENTS].sum(axis=1), axis=0).to_numpy() * 100.0
pred = model.predict(X)
err = np.abs(pred - df["ML_pred_HV"].to_numpy())

assert len(df) == 15, f"Expected 15 candidates, found {len(df)}"
assert list(df["Candidate"]) == list("ABCDEFGHIJKLMNO"), "Candidate labels A–O are not intact"
assert np.max(err) < 1e-6, f"Stored and regenerated predictions differ: max error={np.max(err)}"
assert set(df.loc[(pred >= 450) & (df["HV"] < 450), "Candidate"]) == {"H", "I"}, "Expected false passes H and I"
print("Package validation passed.")
print(f"Maximum prediction discrepancy: {np.max(err):.3e} HV")

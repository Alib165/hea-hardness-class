# Interactive HEA Hardness Practical

A guided Streamlit practical for a Materials Science class. Students combine physical reasoning, machine-learning prediction, experimental validation, and engineering judgement to select an Al–Co–Cr–Cu–Fe–Ni high-entropy alloy.

## Learning sequence

1. **Physical reasoning** — calculate VEC and atomic-size mismatch for a controlled Cu→Al substitution.
2. **Descriptor limits** — show that simple descriptors contain useful physics but do not uniquely determine hardness.
3. **ML screening** — use a trained Extra Trees model to predict Vickers hardness for 15 held-out candidates.
4. **Experimental validation** — reveal measured hardness and identify false passes near the 450-HV design threshold.
5. **Final selection** — choose an experimentally feasible alloy by balancing hardness with a transparent teaching cost index; density and specific hardness are supporting properties.

The 450-HV requirement is a **teaching design threshold**, not a universal industrial requirement.

## Scientific basis

- C. Wen et al., *Machine learning assisted design of high entropy alloys with desired property*, **Acta Materialia 170 (2019) 109–117**. DOI: `10.1016/j.actamat.2019.03.010`.
- U. M. R. Paturi et al., *Evaluating Machine Learning Models for Predicting Hardness of AlCoCrCuFeNi High-Entropy Alloys*, **Crystals 15 (2025) 404**. DOI: `10.3390/cryst15050404`.

The classroom model is deliberately scoped to the **as-cast Al–Co–Cr–Cu–Fe–Ni composition family represented by the data**. It must not be described as a universal hardness predictor.

## Repository structure

```text
hea-hardness-streamlit/
├── app.py
├── requirements.txt
├── README.md
├── INSTRUCTOR_GUIDE.md
├── DEPLOYMENT.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── wen_hea_hardness_155.csv
│   └── classroom_candidates.csv
├── models/
│   └── hea_hardness_model_train140.joblib
└── scripts/
    ├── retrain_model.py
    └── validate_package.py
```

## Run locally

Use Python 3.11+.

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`.

## Validate the package

```bash
python scripts/validate_package.py
```

Expected result:

```text
Package validation passed.
```

To regenerate the 140-row classroom model:

```bash
python scripts/retrain_model.py
```

## Model used in the practical

The supplied model is an `ExtraTreesRegressor` trained on 140 of the 155 compositions, with candidates A–O held out.

Independent model benchmarking performed for the classroom package gave approximately:

- 10-fold CV: **R² = 0.919**, **MAE = 36.9 HV**.
- 15-candidate classroom holdout: **R² = 0.962**, **MAE = 29.8 HV**.

The individual false-pass examples H and I are pedagogically intentional: they show why a strong average metric does not guarantee a safe threshold decision for every material.

## Cost model

The app uses an **illustrative teaching cost index**, not live commodity prices:

| Element | Relative weight |
|---|---:|
| Fe | 1.0 |
| Al | 2.0 |
| Cr | 2.5 |
| Cu | 8.0 |
| Ni | 10.0 |
| Co | 25.0 |

The purpose is to create a transparent competing objective. Replace these weights with a sourced price dataset if real economic conclusions are required.

## Student data/privacy

The app does not require student names, accounts, or personal data. Answers live only in the current Streamlit session unless a student downloads the final recommendation CSV.

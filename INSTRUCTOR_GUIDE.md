# Instructor Guide

## Overall learning aim

Students should leave the activity able to explain why **materials-science descriptors, ML predictions, and experiments play different roles** in a materials-selection workflow.

The exercise is intentionally not framed as “physics versus ML.” Instead:

- physics gives interpretable mechanisms and expectations;
- ML supplies a quantitative mapping where no simple closed-form composition→hardness equation is available;
- experiment checks whether the prediction is actually good enough for the engineering decision;
- students make the final multi-objective judgement.

## Stage 1 — Physical reasoning

**Aim:** show that a controlled compositional change produces a physically interpretable descriptor change.

Correct answers:

- Alloy C: `Co20Cr20Cu20Fe20Ni20`
  - VEC ≈ **8.80**
  - δ ≈ **1.27%**
- Alloy K: `Al20Co20Cr20Fe20Ni20`
  - VEC ≈ **7.20**
  - δ ≈ **5.44%**

Prompt students to explain the direction of the change rather than claiming either descriptor is a hardness equation.

## Stage 2 — Descriptor limits

**Aim:** establish the modelling gap that motivates ML.

Five equiatomic five-element alloys have identical ideal configurational entropy but experimental hardness spans approximately **155–536 HV**.

Expected conclusion:

> VEC and δ contain useful physical information, but neither uniquely determines hardness.

This is the key intellectual bridge to ML.

## Stage 3 — ML screening

**Aim:** use the trained composition→hardness model as a quantitative screening tool.

General model benchmark shown to students:

- 10-fold CV R² ≈ **0.919**
- 10-fold CV MAE ≈ **36.9 HV**

At the 450-HV teaching threshold, the closest predicted passes are:

- **H ≈ 470 HV**
- **I ≈ 470 HV**

Students should identify these as high-risk/borderline decisions before seeing experiments.

## Stage 4 — Experimental validation

**Aim:** demonstrate why average predictive accuracy is not equivalent to guaranteed correct engineering decisions.

False passes:

- H: ML ≈ **470 HV**, experiment = **380 HV**
- I: ML ≈ **470 HV**, experiment = **441 HV**

Both are predicted to satisfy the 450-HV requirement but fail experimentally.

The 15-alloy classroom holdout metrics are approximately:

- MAE = **29.8 HV**
- RMSE = **37.7 HV**
- R² = **0.962**

Use this to emphasize that even an apparently excellent R² can hide consequential threshold errors.

## Stage 5 — Final selection

**Aim:** require students to make an explicit trade-off rather than accept a model ranking as the decision.

Experimentally feasible candidates are **J–O**.

Useful discussion examples:

- **O** maximizes hardness and also has low density/high specific hardness, but is not the cheapest.
- **L** provides substantial hardness with a notably lower teaching cost index.
- **M/N/O** offer progressively high hardness but with different cost implications.

There is no compulsory final winner. Grade the quality of the evidence and trade-off explanation.

## Why density is secondary

Within this particular alloy family, Al often increases hardness while reducing density. Therefore hardness and specific hardness are strongly aligned. Density is worth calculating, but it does not create as strong a conflict as cost. The final decision should therefore emphasize hardness versus cost, with density as supporting information.

## Suggested timing (45–60 minutes)

- 5 min — problem framing
- 10 min — Stage 1 calculations
- 8 min — Stage 2 discussion
- 10 min — Stage 3 ML screening
- 10 min — Stage 4 experimental reveal/discussion
- 10–15 min — Stage 5 group recommendation and short report-back

## What not to say

Avoid statements such as:

- “High entropy makes an alloy hard.”
- “Low VEC means a material will definitely be hard.”
- “Large δ predicts hardness.”
- “R² = 0.92 means 92% accuracy.”
- “The model is valid for all HEAs.”

Prefer:

- “These descriptors capture tendencies.”
- “The ML model interpolates within this experimental alloy family.”
- “Prediction uncertainty matters most near the design threshold.”

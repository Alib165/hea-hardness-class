from __future__ import annotations

import io
import math
import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor


# -----------------------------
# App configuration and constants
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "classroom_candidates.csv"
FULL_DATA_PATH = ROOT / "wen_hea_hardness_155.csv"

ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Ni"]
VEC = {"Al": 3.0, "Co": 9.0, "Cr": 6.0, "Cu": 11.0, "Fe": 8.0, "Ni": 10.0}
RADIUS_PM = {"Al": 143.0, "Co": 125.0, "Cr": 128.0, "Cu": 128.0, "Fe": 126.0, "Ni": 124.0}
ATOMIC_MASS = {"Al": 26.9815, "Co": 58.9332, "Cr": 51.9961, "Cu": 63.546, "Fe": 55.845, "Ni": 58.6934}
DENSITY = {"Al": 2.70, "Co": 8.90, "Cr": 7.19, "Cu": 8.96, "Fe": 7.874, "Ni": 8.908}
# Deliberately pedagogical, not current commodity prices.
COST_WEIGHT = {"Al": 2.0, "Co": 25.0, "Cr": 2.5, "Cu": 8.0, "Fe": 1.0, "Ni": 10.0}
HARDNESS_REQUIREMENT = 450.0

st.set_page_config(
    page_title="HEA Hardness Practical",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 1.4rem; padding-bottom: 3rem;}
    .aim-box {padding: .85rem 1rem; border-left: 5px solid #1F4E78; background: #F3F6F9; border-radius: .35rem; margin-bottom: 1rem;}
    .takeaway {padding: .85rem 1rem; background: #EEF6EE; border: 1px solid #C9DEC9; border-radius: .5rem;}
    .warning-box {padding: .85rem 1rem; background: #FFF7E2; border: 1px solid #E8D39A; border-radius: .5rem;}
    .stage-title {font-size: 1.45rem; font-weight: 700; margin-bottom: .2rem;}
    .small-muted {color:#58636F; font-size:.92rem;}
    div[data-testid="stMetric"] {border:1px solid #E1E5EA; padding:.7rem; border-radius:.55rem; background:#FAFBFC;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Deployment file check
# -----------------------------
_required_files = [DATA_PATH, FULL_DATA_PATH]
_missing_files = [p.name for p in _required_files if not p.exists()]
if _missing_files:
    st.error(
        "The app is missing required repository files: "
        + ", ".join(f"`{name}`" for name in _missing_files)
        + ". Put them in the same GitHub folder as `app.py`."
    )
    st.stop()


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_candidates() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("Candidate").reset_index(drop=True)
    return df


@st.cache_data
def load_full_data() -> pd.DataFrame:
    return pd.read_csv(FULL_DATA_PATH)


@st.cache_resource
def load_model():
    """
    Rebuild the classroom Extra Trees model from source data at app startup.

    Why do this instead of loading a .joblib file?
    ------------------------------------------------
    Serialized scikit-learn models can be sensitive to Python / package-version
    changes in cloud deployment. Re-training here is fast (only 140 rows) and
    exactly reproduces the classroom model because the split and random seed
    are fixed.

    The 15 classroom candidates remain strictly held out from training.
    """
    full = pd.read_csv(FULL_DATA_PATH)
    class_df = pd.read_csv(DATA_PATH)

    # Prefer IDs stored in the classroom-candidate file. The fallback list is
    # the fixed 15-candidate teaching set used when the file has no ID column.
    fallback_holdout_ids = [154, 146, 141, 113, 111, 107, 100, 96, 85, 76, 59, 37, 14, 11, 5]
    if "ID" in class_df.columns:
        holdout_ids = class_df["ID"].astype(int).tolist()
    else:
        holdout_ids = fallback_holdout_ids

    train = full.loc[~full["ID"].astype(int).isin(holdout_ids)].copy()

    if len(train) != 140:
        raise RuntimeError(
            f"Expected 140 training samples after excluding the 15 classroom "
            f"holdouts, but found {len(train)}."
        )

    # Normalize the small rounding deviations in reported atomic percentages.
    X = train[ELEMENTS].div(train[ELEMENTS].sum(axis=1), axis=0).to_numpy() * 100.0
    y = train["HV"].to_numpy(dtype=float)

    model = ExtraTreesRegressor(
        n_estimators=300,
        max_features=1.0,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def normalize_composition(row: pd.Series | dict) -> np.ndarray:
    vals = np.array([float(row[e]) for e in ELEMENTS], dtype=float)
    total = vals.sum()
    if total <= 0:
        raise ValueError("Composition total must be positive.")
    return vals / total * 100.0


def descriptors_from_composition(comp: dict[str, float]) -> dict[str, float]:
    vals = np.array([float(comp[e]) for e in ELEMENTS], dtype=float)
    total = vals.sum()
    if total <= 0:
        raise ValueError("Composition total must be positive.")
    x = {e: float(comp[e]) / total for e in ELEMENTS}
    vec = sum(x[e] * VEC[e] for e in ELEMENTS)
    rbar = sum(x[e] * RADIUS_PM[e] for e in ELEMENTS)
    delta = 100.0 * math.sqrt(sum(x[e] * (1.0 - RADIUS_PM[e] / rbar) ** 2 for e in ELEMENTS))
    mass_term = sum(x[e] * ATOMIC_MASS[e] for e in ELEMENTS)
    volume_term = sum(x[e] * ATOMIC_MASS[e] / DENSITY[e] for e in ELEMENTS)
    rho = mass_term / volume_term
    # cost is mass-fraction weighted
    mass_parts = {e: x[e] * ATOMIC_MASS[e] for e in ELEMENTS}
    mass_total = sum(mass_parts.values())
    cost_score = sum((mass_parts[e] / mass_total) * COST_WEIGHT[e] for e in ELEMENTS)
    return {"VEC": vec, "rbar": rbar, "delta": delta, "density": rho, "cost_score": cost_score}


def predict_hv(row: pd.Series | dict) -> float:
    model = load_model()
    x = normalize_composition(row).reshape(1, -1)
    return float(model.predict(x)[0])


def reset_activity() -> None:
    keep = {}
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.update(keep)
    st.rerun()


def initialise_state() -> None:
    defaults = {
        "stage": 1,
        "s1_complete": False,
        "s2_complete": False,
        "s3_complete": False,
        "s4_complete": False,
        "s5_complete": False,
        "experimental_revealed": False,
        "stage2_prediction_locked": False,
        "stage2_prediction_order": [],
        "final_choice": None,
        "final_priority": None,
        "final_reason": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def go_to(stage: int) -> None:
    st.session_state.stage = stage
    st.rerun()


def progress_header(stage: int) -> None:
    labels = [
        "1 · Physical reasoning",
        "2 · Descriptor limits",
        "3 · ML screening",
        "4 · Experimental validation",
        "5 · Final selection",
    ]
    st.progress((stage - 1) / 4 if stage > 1 else 0.02)
    st.caption("  →  ".join([f"**{x}**" if i + 1 == stage else x for i, x in enumerate(labels)]))


def stage_nav(current: int) -> None:
    cols = st.columns([1, 1, 5])
    with cols[0]:
        if current > 1 and st.button("← Previous", use_container_width=True):
            go_to(current - 1)
    with cols[1]:
        if current < 5:
            completed = bool(st.session_state.get(f"s{current}_complete", False))
            if st.button("Next →", disabled=not completed, use_container_width=True):
                go_to(current + 1)


def controlled_five_alloy_table(full_df: pd.DataFrame) -> pd.DataFrame:
    # Exact five equiatomic examples in the 155-row dataset.
    configs = [
        ("Al", {"Al": 0, "Co": 20, "Cr": 20, "Cu": 20, "Fe": 20, "Ni": 20}),
        ("Fe", {"Al": 20, "Co": 20, "Cr": 20, "Cu": 20, "Fe": 0, "Ni": 20}),
        ("Co", {"Al": 20, "Co": 0, "Cr": 20, "Cu": 20, "Fe": 20, "Ni": 20}),
        ("Cu", {"Al": 20, "Co": 20, "Cr": 20, "Cu": 0, "Fe": 20, "Ni": 20}),
        ("Cr", {"Al": 20, "Co": 20, "Cr": 0, "Cu": 20, "Fe": 20, "Ni": 20}),
    ]
    rows = []
    for omitted, comp in configs:
        desc = descriptors_from_composition(comp)
        mask = np.ones(len(full_df), dtype=bool)
        for e in ELEMENTS:
            mask &= np.isclose(full_df[e].astype(float).values, comp[e], atol=0.25)
        match = full_df.loc[mask]
        hv = float(match.iloc[0]["HV"]) if len(match) else np.nan
        composition_label = " ".join(
            f"{e}{int(comp[e])}" for e in ELEMENTS if comp[e] > 0
        )
        rows.append(
            {
                "Element omitted": omitted,
                "Real dataset composition": composition_label,
                "VEC": desc["VEC"],
                "δ (%)": desc["delta"],
                "Density (g/cm³)": desc["density"],
                "Experimental HV": hv,
            }
        )
    return pd.DataFrame(rows)


def teaching_cost_index(df: pd.DataFrame) -> pd.Series:
    raw = []
    for _, row in df.iterrows():
        comp = {e: float(row[e]) for e in ELEMENTS}
        raw.append(descriptors_from_composition(comp)["cost_score"])
    raw = np.array(raw)
    return pd.Series(raw / raw.min(), index=df.index)


def prediction_table(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["ML HV"] = [predict_hv(r) for _, r in out.iterrows()]
    out["ML screen"] = np.where(out["ML HV"] >= HARDNESS_REQUIREMENT, "PASS", "Fail")
    return out


# -----------------------------
# Initialize and common UI
# -----------------------------
initialise_state()
candidates = load_candidates()
full_df = load_full_data()

with st.sidebar:
    st.header("HEA hardness practical")
    st.write("**Design requirement:** ≥ 450 HV")
    st.caption("The 450-HV threshold is a teaching design requirement, not a universal industrial standard.")
    st.divider()
    st.write("**Current stage**")
    st.write(f"{st.session_state.stage} of 5")
    st.write("1. Physical reasoning")
    st.write("2. Why descriptors are insufficient")
    st.write("3. ML screening")
    st.write("4. Experimental validation")
    st.write("5. Final material selection")
    st.divider()
    if st.button("Restart activity", use_container_width=True):
        reset_activity()
    with st.expander("Scientific scope"):
        st.write(
            "The ML model is intended for interpolation within the as-cast Al–Co–Cr–Cu–Fe–Ni composition family represented by the training data. "
            "It is not a universal hardness model."
        )

st.title("Interactive materials selection: hardness of high-entropy alloys")
st.write(
    "You will follow one scientific question at a time: first build a physical expectation from composition, "
    "then identify the limit of the simple descriptors, then use ML for quantitative prediction, "
    "then test ML against experiment, and finally make a material-selection decision."
)
st.caption(
    "All controlled-comparison alloys and all 15 classroom candidates used in the activity are real samples from the experimental dataset; the model outputs are predictions."
)
progress_header(st.session_state.stage)


# -----------------------------
# Stage 1
# -----------------------------
if st.session_state.stage == 1:
    st.markdown('<div class="stage-title">Stage 1 — Build a physical expectation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aim-box"><strong>Aim:</strong> use composition to calculate two simple materials descriptors — VEC and atomic-size mismatch — before seeing any hardness prediction.</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "We begin with **two real experimental samples from the 155-alloy dataset**. "
        "Both are equiatomic five-element alloys: every element that is present is **20 at.% = atomic fraction 0.20**. "
        "The controlled chemistry change is **Cu → Al**. You will calculate the descriptors step by step before seeing hardness."
    )
    st.info(
        "Why only two samples here? This first step is for learning the calculation cleanly. "
        "In Stage 2 you will immediately expand the comparison to five real controlled alloys."
    )

    st.subheader("1. Information you need")
    st.write("Use the following constants. Everything required for the calculation is shown here.")
    const = pd.DataFrame(
        {
            "Element": ELEMENTS,
            "VEC value": [VEC[e] for e in ELEMENTS],
            "Atomic radius, rᵢ (pm)": [RADIUS_PM[e] for e in ELEMENTS],
        }
    )
    st.dataframe(const, hide_index=True, use_container_width=True)

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("**VEC**")
        st.latex(r"VEC=\sum_i x_i VEC_i")
        st.caption("Multiply each atomic fraction by its elemental VEC value, then add the terms.")
    with e2:
        st.markdown("**Mean atomic radius**")
        st.latex(r"\bar r=\sum_i x_i r_i")
        st.caption("You need r̄ before you can calculate atomic-size mismatch.")
    with e3:
        st.markdown("**Atomic-size mismatch**")
        st.latex(r"\delta=100\sqrt{\sum_i x_i\left(1-\frac{r_i}{\bar r}\right)^2}")
        st.caption("Use atomic fractions, not percentages, in the summation.")

    st.markdown(
        '<div class="warning-box"><strong>Worked method, not the answer:</strong> if an element is 20 at.%, use x = 0.20. For VEC, one term looks like 0.20 × VEC<sub>element</sub>. For r̄, one term looks like 0.20 × r<sub>element</sub>. Add the five terms that are present in the alloy.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
        **What each calculation is for**

        - **VEC** gives a simple electronic-structure / phase-tendency descriptor.
        - **Mean radius \\(\bar r\\)** is an intermediate quantity needed for \\(\delta\\).
        - **Atomic-size mismatch \(\delta\)** measures how different the constituent atomic sizes are and therefore gives a simple indication of lattice mismatch/distortion.
        - **None of these equations calculates hardness directly.** Their role is to build a physically informed expectation.
        """
    )

    st.subheader("2. Calculate Alloy C")
    st.code("Co20 Cr20 Cu20 Fe20 Ni20", language=None)
    st.caption("Al is absent. Co, Cr, Cu, Fe and Ni are each 20 at.% = 0.20 atomic fraction.")
    c1, c2, c3 = st.columns(3)
    with c1:
        c_vec = st.number_input("Alloy C: VEC", min_value=0.0, max_value=12.0, step=0.01, key="c_vec")
    with c2:
        c_rbar = st.number_input("Alloy C: mean radius r̄ (pm)", min_value=100.0, max_value=160.0, step=0.01, key="c_rbar")
    with c3:
        c_delta = st.number_input("Alloy C: δ (%)", min_value=0.0, max_value=20.0, step=0.01, key="c_delta")

    st.subheader("3. Calculate Alloy K")
    st.code("Al20 Co20 Cr20 Fe20 Ni20", language=None)
    st.caption("Cu is absent. Al, Co, Cr, Fe and Ni are each 20 at.% = 0.20 atomic fraction.")
    k1, k2, k3 = st.columns(3)
    with k1:
        k_vec = st.number_input("Alloy K: VEC", min_value=0.0, max_value=12.0, step=0.01, key="k_vec")
    with k2:
        k_rbar = st.number_input("Alloy K: mean radius r̄ (pm)", min_value=100.0, max_value=160.0, step=0.01, key="k_rbar")
    with k3:
        k_delta = st.number_input("Alloy K: δ (%)", min_value=0.0, max_value=20.0, step=0.01, key="k_delta")

    st.markdown("**Before checking:** based only on your VEC and δ values, which alloy would you expect to show greater resistance to local plastic deformation?")
    expectation = st.radio(
        "Your physical expectation",
        ["Choose one…", "Alloy C", "Alloy K", "There is no reason to expect a difference"],
        key="stage1_expectation",
        horizontal=True,
    )

    if st.button("Check my calculations", type="primary"):
        ok = (
            abs(c_vec - 8.80) <= 0.05
            and abs(c_rbar - 126.20) <= 0.08
            and abs(c_delta - 1.27) <= 0.08
            and abs(k_vec - 7.20) <= 0.05
            and abs(k_rbar - 129.20) <= 0.08
            and abs(k_delta - 5.44) <= 0.08
        )
        if expectation == "Choose one…":
            st.error("Make a physical expectation before checking. The point is to commit to a materials-science hypothesis before seeing hardness data.")
        elif not ok:
            st.error("Not quite. Check each step in order: atomic fraction → VEC; atomic fraction → mean radius r̄; then use r̄ in the δ equation.")
        else:
            st.session_state.s1_complete = True
            st.success(
                "Correct. Alloy C: VEC ≈ 8.80, r̄ ≈ 126.20 pm, δ ≈ 1.27%. "
                "Alloy K: VEC ≈ 7.20, r̄ ≈ 129.20 pm, δ ≈ 5.44%."
            )

    if st.session_state.s1_complete:
        st.markdown(
            """
            <div class="takeaway"><strong>Stage 1 output:</strong> replacing Cu with Al lowers VEC strongly and increases atomic-size mismatch strongly. These descriptors give you a physical basis for expecting different deformation resistance, but they still do <em>not</em> calculate Vickers hardness.</div>
            """,
            unsafe_allow_html=True,
        )
    stage_nav(1)


# -----------------------------
# Stage 2
# -----------------------------
elif st.session_state.stage == 2:
    st.markdown('<div class="stage-title">Stage 2 — Test the limits of the simple descriptors</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aim-box"><strong>Aim:</strong> use VEC and atomic-size mismatch to make a physically informed prediction, then compare that prediction with real hardness data and identify where the simple descriptors stop being sufficient.</div>',
        unsafe_allow_html=True,
    )

    table = controlled_five_alloy_table(full_df)
    st.write(
        "Now expand from the two hand-calculated examples to **five real experimental samples from the same dataset**. "
        "Every alloy contains five elements at equal fractions; a different element is omitted in each case. "
        "Because the five alloys are all equiatomic, their ideal configurational entropy is the same."
    )

    st.subheader("1. Make a prediction before seeing hardness")
    st.write(
        "For now, the experimental hardness values are deliberately hidden. Use only **VEC**, **δ**, and your materials-science reasoning. "
        "Rank the five alloys from the one you expect to be **softest** to the one you expect to be **hardest**."
    )

    descriptor_view = table[["Element omitted", "Real dataset composition", "VEC", "δ (%)", "Density (g/cm³)"]].copy()
    st.dataframe(
        descriptor_view.style.format({"VEC": "{:.2f}", "δ (%)": "{:.2f}", "Density (g/cm³)": "{:.2f}"}),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "How to reason: lower VEC and larger δ may suggest greater resistance to plastic deformation in this alloy family, "
        "but these are tendencies — not hardness equations."
    )

    labels = table["Element omitted"].tolist()
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        r1 = st.selectbox("1 · Expected softest", ["Choose…"] + labels, key="rank1")
    with c2:
        r2 = st.selectbox("2", ["Choose…"] + labels, key="rank2")
    with c3:
        r3 = st.selectbox("3", ["Choose…"] + labels, key="rank3")
    with c4:
        r4 = st.selectbox("4", ["Choose…"] + labels, key="rank4")
    with c5:
        r5 = st.selectbox("5 · Expected hardest", ["Choose…"] + labels, key="rank5")

    if not st.session_state.stage2_prediction_locked:
        if st.button("Lock my prediction and reveal the experimental hardness", type="primary"):
            order = [r1, r2, r3, r4, r5]
            if "Choose…" in order:
                st.error("Complete all five ranking positions before revealing the experimental results.")
            elif len(set(order)) != 5:
                st.error("Use each omitted-element alloy exactly once in your ranking.")
            else:
                st.session_state.stage2_prediction_order = order
                st.session_state.stage2_prediction_locked = True
                st.rerun()

    if st.session_state.stage2_prediction_locked:
        predicted = " → ".join(st.session_state.stage2_prediction_order)
        actual_order = (
            table.sort_values("Experimental HV")["Element omitted"].tolist()
        )
        actual = " → ".join(actual_order)

        st.markdown(
            f'<div class="warning-box"><strong>Your prediction, softest → hardest:</strong> {predicted}</div>',
            unsafe_allow_html=True,
        )

        st.subheader("2. Reveal the real experimental result")
        st.dataframe(
            table.style.format({"VEC": "{:.2f}", "δ (%)": "{:.2f}", "Density (g/cm³)": "{:.2f}", "Experimental HV": "{:.0f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.write(f"**Actual ranking, softest → hardest:** {actual}")

        st.info(
            "Hardness spans roughly 155–536 HV even though ideal configurational entropy is identical for all five alloys. "
            "The broad VEC/δ trends are useful, but they do not uniquely determine the ranking."
        )

        st.markdown(
            r"""
            ### What should you notice?

            **Configurational entropy:** all five alloys have the same ideal configurational entropy, yet their hardness is very different. Entropy alone cannot explain hardness.

            **VEC:** the Al-free alloy has the highest VEC and the lowest hardness, while several lower-VEC alloys are much harder. This is a useful trend — but not a one-to-one rule.

            **Atomic-size mismatch:** the Al-free alloy has a very small \(\delta\), while the Al-containing alloys have much larger \(\delta\). Again, this is physically informative, but alloys with nearly the same \(\delta\) still show different hardness values.

            Therefore the descriptors are doing something useful: **they reveal physical trends. But they are not enough to calculate a unique hardness value.**
            """
        )

        conclusion = st.radio(
            "Which conclusion is best supported by this comparison?",
            [
                "Choose one…",
                "Configurational entropy is sufficient to calculate hardness.",
                "VEC and δ contain useful physical information, but neither uniquely determines hardness.",
                "Atomic-size mismatch alone determines hardness.",
            ],
            key="stage2_conclusion",
        )
        if st.button("Check my Stage 2 conclusion", type="primary"):
            if conclusion == "VEC and δ contain useful physical information, but neither uniquely determines hardness.":
                st.session_state.s2_complete = True
                st.success(
                    "Exactly. Your hand calculations describe meaningful materials physics, but a multivariable composition–hardness mapping remains. "
                    "That is the specific gap the ML model will address next."
                )
            else:
                st.error(
                    "Look again: alloys with similar δ still have different hardness, and configurational entropy is identical across the whole comparison."
                )

    if st.session_state.s2_complete:
        st.markdown(
            '<div class="takeaway"><strong>Stage 2 output:</strong> you first committed to a physics-based prediction, then tested it against real measurements. Simple descriptors help us understand the trend, but they do not provide a unique quantitative hardness prediction. This is where ML has a legitimate role.</div>',
            unsafe_allow_html=True,
        )
    stage_nav(2)


# -----------------------------
# Stage 3
# -----------------------------
elif st.session_state.stage == 3:
    st.markdown('<div class="stage-title">Stage 3 — Use ML for quantitative screening</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aim-box"><strong>Aim:</strong> use a data-driven model to estimate hardness for 15 unseen classroom candidates and identify predictions that are risky because they sit close to the design threshold.</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "The classroom Extra Trees model was trained on **140 experimental alloy compositions**. The 15 candidates below were held out from that training set. "
        "The model receives only Al, Co, Cr, Cu, Fe and Ni atomic percentages."
    )
    st.markdown(
        r"""
        **Why ML appears only now:**

        In Stages 1–2 you established that composition-based descriptors such as VEC and \\(\delta\\) contain real physical information, but do not uniquely determine hardness. ML is therefore being used for a task the hand calculations cannot already solve:

        \[
        [Al,Co,Cr,Cu,Fe,Ni] \longrightarrow \widehat{HV}
        \]

        The ML output is a **prediction**, not an experimental fact.
        """
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("10-fold CV R²", "0.919")
    m2.metric("10-fold CV MAE", "36.9 HV")
    m3.metric("Design threshold", "450 HV")
    st.caption("The cross-validation numbers characterize average predictive performance; they do not guarantee every individual decision will be correct.")

    pred = prediction_table(candidates)
    view = pred[["Candidate", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "VEC", "atomic_size_mismatch_pct", "ML HV", "ML screen"]].copy()
    view = view.rename(columns={"atomic_size_mismatch_pct": "δ (%)"})
    st.dataframe(
        view.style.format({e: "{:.1f}" for e in ELEMENTS} | {"VEC": "{:.2f}", "δ (%)": "{:.2f}", "ML HV": "{:.0f}"}),
        hide_index=True,
        use_container_width=True,
    )

    trust = st.radio(
        "Would you trust all ML-predicted passes equally?",
        [
            "Choose one…",
            "Yes — if the model predicts a pass, the distance from the threshold does not matter.",
            "No — predictions close to 450 HV are riskier because normal model error could move the real value below the requirement.",
        ],
        key="stage3_trust",
    )

    passers = pred.loc[pred["ML HV"] >= HARDNESS_REQUIREMENT, "Candidate"].tolist()
    chosen = st.multiselect(
        "Select the TWO predicted passes that are closest to 450 HV and therefore deserve the most caution.",
        passers,
        key="borderline_selection",
        max_selections=2,
    )
    if st.button("Check my Stage 3 reasoning", type="primary"):
        trust_ok = trust.startswith("No —")
        border_ok = set(chosen) == {"H", "I"}
        if trust == "Choose one…":
            st.error("First decide whether all predicted passes should be trusted equally.")
        elif not trust_ok:
            st.error("A prediction margin matters. A material predicted only slightly above the requirement is more vulnerable to normal prediction error.")
        elif not border_ok:
            st.error("Now identify the two ML predictions that sit only slightly above 450 HV.")
        else:
            st.session_state.s3_complete = True
            st.success(
                "Correct. H and I are only about 20 HV above the requirement. Their margin is smaller than the model's typical MAE scale, "
                "so they should be treated as risky predicted passes rather than as equally secure as candidates far above 450 HV."
            )

    if st.session_state.s3_complete:
        st.markdown(
            '<div class="takeaway"><strong>Stage 3 output:</strong> ML creates a quantitative feasibility screen, but a pass/fail label is not enough. Prediction margin matters: H and I are borderline, while candidates far above 450 HV have a much larger decision margin.</div>',
            unsafe_allow_html=True,
        )
    stage_nav(3)


# -----------------------------
# Stage 4
# -----------------------------
elif st.session_state.stage == 4:
    st.markdown('<div class="stage-title">Stage 4 — Challenge ML with experiment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aim-box"><strong>Aim:</strong> determine whether the ML-based pass/fail decisions survive experimental validation.</div>',
        unsafe_allow_html=True,
    )

    pred = prediction_table(candidates)
    if not st.session_state.experimental_revealed:
        st.write("The experimental Vickers-hardness measurements have been withheld until this point.")
        if st.button("Reveal experimental results", type="primary"):
            st.session_state.experimental_revealed = True
            st.rerun()
    else:
        out = pred[["Candidate", "ML HV", "HV"]].copy()
        out["ML decision"] = np.where(out["ML HV"] >= HARDNESS_REQUIREMENT, "PASS", "Fail")
        out["Experimental decision"] = np.where(out["HV"] >= HARDNESS_REQUIREMENT, "PASS", "Fail")
        out["Error (ML − exp.)"] = out["ML HV"] - out["HV"]
        out = out.rename(columns={"HV": "Experimental HV"})
        st.dataframe(
            out.style.format({"ML HV": "{:.0f}", "Experimental HV": "{:.0f}", "Error (ML − exp.)": "{:+.0f}"}),
            hide_index=True,
            use_container_width=True,
        )

        ml_pass = pred.loc[pred["ML HV"] >= HARDNESS_REQUIREMENT, "Candidate"].tolist()
        false_choice = st.multiselect(
            "Which candidates are FALSE PASSES: predicted to pass, but experimentally below 450 HV?",
            ml_pass,
            key="false_pass_selection",
            max_selections=2,
        )
        if st.button("Check false passes", type="primary"):
            if set(false_choice) == {"H", "I"}:
                st.session_state.s4_complete = True
                st.success("Correct. H and I are false passes: ML says PASS, experiment says Fail.")
            else:
                st.error("A false pass must have ML HV ≥ 450 but experimental HV < 450.")

        if st.session_state.s4_complete:
            h1, h2, h3 = st.columns(3)
            h1.metric("15-alloy holdout MAE", "29.8 HV")
            h2.metric("15-alloy holdout RMSE", "37.7 HV")
            h3.metric("15-alloy holdout R²", "0.962")
            st.markdown(
                '<div class="warning-box"><strong>Stage 4 output:</strong> a model can have strong average statistics and still make a consequential individual engineering error. Validation matters most near decision boundaries.</div>',
                unsafe_allow_html=True,
            )
    stage_nav(4)


# -----------------------------
# Stage 5
# -----------------------------
elif st.session_state.stage == 5:
    st.markdown('<div class="stage-title">Stage 5 — Make the final materials-selection decision</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aim-box"><strong>Aim:</strong> select one experimentally feasible alloy by balancing hardness with material cost; use density and specific hardness as supporting information.</div>',
        unsafe_allow_html=True,
    )

    feasible = candidates.loc[candidates["HV"] >= HARDNESS_REQUIREMENT].copy()
    feasible["Relative cost"] = teaching_cost_index(candidates).loc[feasible.index].values
    feasible["Specific hardness"] = feasible["HV"] / feasible["rho_mix_g_cm3"]

    st.subheader("How the selection quantities are calculated")
    st.write(
        "You do not need any hidden information. The elemental constants and equations used for density and the cost index are shown below."
    )
    st.markdown(
        r"""
        **What each final-selection quantity is doing**

        - **Experimental hardness:** first checks whether the alloy actually satisfies the 450-HV requirement.
        - **Density:** tells you about mass; in this particular dataset many Al-rich hard alloys are also relatively light, so density may reinforce rather than reverse the hardness ranking.
        - **Specific hardness \\(HV/\rho\\):** shows hardness relative to density; treat it as a simple teaching comparison, not a universal materials index.
        - **Relative cost:** creates the clearest competing design objective here. A harder alloy may not be worth a large increase in material cost.

        The goal is therefore **not** to maximise every column at once. It is to make the trade-off explicit and defend a decision.
        """
    )
    selection_constants = pd.DataFrame({
        "Element": ELEMENTS,
        "Atomic mass Mᵢ (g/mol)": [ATOMIC_MASS[e] for e in ELEMENTS],
        "Element density ρᵢ (g/cm³)": [DENSITY[e] for e in ELEMENTS],
        "Teaching cost weight Cᵢ": [COST_WEIGHT[e] for e in ELEMENTS],
    })
    st.dataframe(selection_constants, hide_index=True, use_container_width=True)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("**Ideal mixture density**")
        st.latex(r"\rho_{alloy}=\frac{\sum_i x_iM_i}{\sum_i x_iM_i/\rho_i}")
        st.caption("xᵢ is atomic fraction, Mᵢ atomic mass, and ρᵢ elemental density.")
    with dc2:
        st.markdown("**Relative raw-material cost**")
        st.latex(r"w_i=\frac{x_iM_i}{\sum_j x_jM_j},\qquad C_{raw}=\sum_i w_iC_i")
        st.caption("Atomic fractions are first converted to mass fractions wᵢ. The displayed relative cost is then normalized to the cheapest classroom candidate.")

    st.markdown(
        '<div class="warning-box"><strong>Cost note:</strong> Fe=1, Al=2, Cr=2.5, Cu=8, Ni=10 and Co=25 are deliberately transparent teaching weights, not current commodity prices. The aim is to create a competing design objective, not to estimate procurement cost.</div>',
        unsafe_allow_html=True,
    )

    sort_label = st.selectbox(
        "Sort the experimentally feasible candidates by:",
        ["Highest hardness", "Lowest relative cost", "Highest specific hardness", "Lowest density"],
    )
    if sort_label == "Highest hardness":
        feasible = feasible.sort_values("HV", ascending=False)
    elif sort_label == "Lowest relative cost":
        feasible = feasible.sort_values("Relative cost", ascending=True)
    elif sort_label == "Highest specific hardness":
        feasible = feasible.sort_values("Specific hardness", ascending=False)
    else:
        feasible = feasible.sort_values("rho_mix_g_cm3", ascending=True)

    final_view = feasible[["Candidate", "HV", "rho_mix_g_cm3", "Specific hardness", "Relative cost"]].rename(
        columns={"HV": "Experimental HV", "rho_mix_g_cm3": "Density (g/cm³)"}
    )
    st.dataframe(
        final_view.style.format(
            {"Experimental HV": "{:.0f}", "Density (g/cm³)": "{:.2f}", "Specific hardness": "{:.1f}", "Relative cost": "{:.2f}×"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.write("There is deliberately **no universal weighted score**. State what you value and defend the trade-off.")
    c1, c2 = st.columns(2)
    with c1:
        final_choice = st.selectbox("Your recommended candidate", feasible["Candidate"].tolist(), key="final_candidate_widget")
    with c2:
        priority = st.selectbox(
            "Your main priority",
            ["Balanced compromise", "Maximum hardness", "Lowest raw-material cost", "Highest hardness-to-weight ratio"],
            key="final_priority_widget",
        )
    reason = st.text_area(
        "Justify your recommendation using at least TWO pieces of evidence from the table.",
        height=130,
        placeholder="Example: Candidate L meets the 450-HV requirement and costs much less than the harder Co-rich candidates, while still providing strong specific hardness...",
        key="final_reason_widget",
    )

    if st.button("Submit final recommendation", type="primary"):
        if len(reason.strip()) < 35:
            st.error("Please give a short evidence-based justification using at least two considerations.")
        else:
            st.session_state.s5_complete = True
            st.session_state.final_choice = final_choice
            st.session_state.final_priority = priority
            st.session_state.final_reason = reason.strip()

    if st.session_state.s5_complete:
        row = feasible.loc[feasible["Candidate"] == st.session_state.final_choice].iloc[0]
        st.success("Recommendation recorded. Different groups can reach different defensible answers if the trade-off is justified.")
        st.markdown(
            f"""
            <div class="takeaway">
            <strong>Your recommendation: Candidate {st.session_state.final_choice}</strong><br>
            Priority: {st.session_state.final_priority}<br>
            Experimental hardness: {row['HV']:.0f} HV · Density: {row['rho_mix_g_cm3']:.2f} g/cm³ · Specific hardness: {row['HV']/row['rho_mix_g_cm3']:.1f} · Relative cost: {row['Relative cost']:.2f}×<br><br>
            <strong>Your justification:</strong> {html.escape(st.session_state.final_reason)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_df = pd.DataFrame(
            [
                {
                    "recommended_candidate": st.session_state.final_choice,
                    "priority": st.session_state.final_priority,
                    "justification": st.session_state.final_reason,
                    "experimental_HV": float(row["HV"]),
                    "density_g_cm3": float(row["rho_mix_g_cm3"]),
                    "specific_hardness": float(row["HV"] / row["rho_mix_g_cm3"]),
                    "relative_cost": float(row["Relative cost"]),
                }
            ]
        )
        st.download_button(
            "Download my recommendation (.csv)",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="hea_hardness_recommendation.csv",
            mime="text/csv",
        )

        with st.expander("Optional extension — test your own composition"):
            st.write(
                "Explore the trained model after completing the practical. Enter any composition within the same six-element family. "
                "Treat predictions far outside the training-data distribution with caution."
            )
            cols = st.columns(3)
            custom = {}
            defaults = {"Al": 20.0, "Co": 20.0, "Cr": 20.0, "Cu": 0.0, "Fe": 20.0, "Ni": 20.0}
            for i, e in enumerate(ELEMENTS):
                with cols[i % 3]:
                    custom[e] = st.number_input(f"{e} at.%", min_value=0.0, max_value=100.0, value=defaults[e], step=1.0, key=f"custom_{e}")
            total = sum(custom.values())
            st.caption(f"Current total: {total:.1f} at.% (the app normalizes the composition before prediction).")
            if total > 0:
                desc = descriptors_from_composition(custom)
                pred_hv = predict_hv(custom)
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Predicted hardness", f"{pred_hv:.0f} HV")
                e2.metric("VEC", f"{desc['VEC']:.2f}")
                e3.metric("δ", f"{desc['delta']:.2f}%")
                e4.metric("Density", f"{desc['density']:.2f} g/cm³")
                st.caption("This extension is exploratory. Do not interpret it as experimental validation or as reliable extrapolation outside the training domain.")

    stage_nav(5)


st.divider()
st.caption(
    "Scientific basis: Wen et al., Acta Materialia 170 (2019) 109–117, DOI 10.1016/j.actamat.2019.03.010. "
    "Independent hardness-ML study: Paturi et al., Crystals 15 (2025) 404, DOI 10.3390/cryst15050404."
)


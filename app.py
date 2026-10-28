from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA

# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HEA Hardness — 60-minute Group Practical",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
FULL_DATA_PATH = ROOT / "wen_hea_hardness_155.csv"

# Allow either the GitHub classroom filename or the original instructor export.
CANDIDATE_FILENAMES = [
    "classroom_candidates.csv",
    "hea_hardness_15_candidates_instructor.csv",
    "15_classroom_candidates___instructor_view.csv",
]

ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Ni"]
HARDNESS_REQUIREMENT = 450.0

VEC = {"Al": 3.0, "Co": 9.0, "Cr": 6.0, "Cu": 11.0, "Fe": 8.0, "Ni": 10.0}
RADIUS_PM = {"Al": 143.0, "Co": 125.0, "Cr": 128.0, "Cu": 128.0, "Fe": 126.0, "Ni": 124.0}
ATOMIC_MASS = {"Al": 26.9815, "Co": 58.9332, "Cr": 51.9961, "Cu": 63.546, "Fe": 55.845, "Ni": 58.6934}
ELEMENT_DENSITY = {"Al": 2.700, "Co": 8.900, "Cr": 7.190, "Cu": 8.960, "Fe": 7.874, "Ni": 8.908}

# Transparent teaching weights only — not market prices.
COST_WEIGHT = {"Al": 2.0, "Co": 25.0, "Cr": 2.5, "Cu": 8.0, "Fe": 1.0, "Ni": 10.0}

# Fixed 15-sample classroom holdout used to build the original model.
FALLBACK_HOLDOUT_IDS = [154, 146, 141, 113, 111, 107, 100, 96, 85, 76, 59, 37, 14, 11, 5]

STAGE_NAMES = [
    "1 · Physical descriptors",
    "2 · Research + hypothesis",
    "3 · Interactive ML laboratory",
    "4 · Experimental validation",
    "5 · Materials-selection challenge",
]

STAGE_TIMES = ["12 min", "10 min", "16 min", "7 min", "15 min"]

st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 1.2rem; padding-bottom: 3rem;}
.aim-box {padding:.9rem 1rem; border-left:5px solid #1F4E78; background:#F4F7FA; border-radius:.4rem; margin:.6rem 0 1rem 0;}
.takeaway {padding:.9rem 1rem; border-left:5px solid #2E7D32; background:#F2F8F2; border-radius:.4rem; margin:.8rem 0;}
.warning-box {padding:.9rem 1rem; border-left:5px solid #B7791F; background:#FFF8E8; border-radius:.4rem; margin:.8rem 0;}
.research-box {padding:.9rem 1rem; border-left:5px solid #6B46C1; background:#F8F5FF; border-radius:.4rem; margin:.8rem 0;}
.role-box {padding:.75rem .9rem; border:1px solid #D8DEE6; background:#FBFCFD; border-radius:.45rem; margin:.4rem 0;}
.stage-title {font-size:1.5rem; font-weight:700; margin-bottom:.15rem;}
.small-muted {font-size:.91rem; color:#5B6570;}
div[data-testid="stMetric"] {border:1px solid #E1E5EA; padding:.65rem; border-radius:.5rem; background:#FBFCFD;}

.dist-card {border:1px solid #E1E5EA; border-radius:.65rem; padding:.72rem .78rem; background:#FFFFFF; margin:.25rem 0 .65rem 0; min-height:165px;}
.dist-head {display:flex; justify-content:space-between; align-items:baseline; margin-bottom:.35rem;}
.dist-element {font-size:1.08rem; font-weight:750; color:#1F2937;}
.dist-value {font-size:.88rem; font-weight:650; color:#1F4E78;}
.mini-hist {height:62px; display:flex; align-items:flex-end; gap:3px; padding:4px 2px 2px 2px; border-bottom:1px solid #CBD5E1; margin:.2rem 0 .35rem 0;}
.mini-bar {flex:1; min-width:4px; background:#AFC7DD; border-radius:3px 3px 0 0;}
.dist-stats {font-size:.78rem; color:#64748B; line-height:1.35;}
.range-card {border:1px solid #E1E5EA; border-radius:.65rem; padding:.72rem .8rem; background:#FFFFFF; margin:.25rem 0 .65rem 0;}
.range-head {display:flex; justify-content:space-between; align-items:center; margin-bottom:.45rem;}
.range-element {font-size:1.05rem; font-weight:750; color:#1F2937;}
.range-query {font-size:.95rem; font-weight:750; color:#1F4E78;}
.range-track {position:relative; height:18px; background:#F1F5F9; border-radius:999px; overflow:visible; margin:.55rem .15rem .35rem .15rem;}
.range-observed {position:absolute; top:6px; height:6px; background:#CBD5E1; border-radius:999px;}
.range-typical {position:absolute; top:4px; height:10px; background:#90B6D8; border-radius:999px;}
.range-median {position:absolute; top:1px; width:2px; height:16px; background:#334155;}
.range-query-marker {position:absolute; top:-3px; width:4px; height:24px; background:#B42318; border-radius:3px; transform:translateX(-2px);}
.range-axis {display:flex; justify-content:space-between; font-size:.68rem; color:#94A3B8; margin:0 .05rem;}
.range-note {font-size:.77rem; color:#64748B; line-height:1.35; margin-top:.35rem;}
.status-pill {display:inline-block; padding:.28rem .6rem; border-radius:999px; font-size:.78rem; font-weight:750; letter-spacing:.01em;}
.status-low {background:#EAF7ED; color:#216E39;}
.status-medium {background:#FFF4DB; color:#8A5A00;}
.status-high {background:#FDECEC; color:#A61B1B;}
.coverage-panel {border:1px solid #D8DEE6; border-radius:.8rem; padding:1rem 1.05rem; background:#FBFCFD; margin:.25rem 0 .8rem 0;}
.coverage-title {font-size:1.08rem; font-weight:780; color:#1F2937; margin-bottom:.4rem;}
.coverage-big {font-size:1.25rem; font-weight:800; margin:.2rem 0 .45rem 0;}
.coverage-explain {font-size:.9rem; color:#475569; line-height:1.5;}
.distance-track {position:relative; height:16px; background:#EEF2F6; border-radius:999px; margin:.8rem .15rem .35rem .15rem; overflow:visible;}
.distance-safe {position:absolute; left:0; top:0; bottom:0; background:#DCEFE1; border-radius:999px 0 0 999px;}
.distance-edge {position:absolute; top:0; bottom:0; background:#FBE7B7;}
.distance-marker {position:absolute; top:-4px; width:4px; height:24px; background:#1F2937; border-radius:3px; transform:translateX(-2px);}
.distance-labels {display:flex; justify-content:space-between; font-size:.72rem; color:#64748B;}
.legend-line {display:flex; gap:.8rem; flex-wrap:wrap; font-size:.74rem; color:#64748B; margin-top:.25rem;}
.legend-key {display:inline-flex; align-items:center; gap:.3rem;}
.legend-swatch {width:14px; height:7px; border-radius:4px; display:inline-block;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# FILES / DATA / MODEL
# ============================================================


def find_candidate_path() -> Path | None:
    for name in CANDIDATE_FILENAMES:
        p = ROOT / name
        if p.exists():
            return p
    return None


CANDIDATE_PATH = find_candidate_path()

if not FULL_DATA_PATH.exists():
    st.error(
        "Required file `wen_hea_hardness_155.csv` is missing. Put it in the same GitHub folder as `app.py`."
    )
    st.stop()

if CANDIDATE_PATH is None:
    st.error(
        "The 15-candidate classroom CSV is missing. Put `classroom_candidates.csv` in the same GitHub folder as `app.py`."
    )
    st.stop()


@st.cache_data
def load_full_data() -> pd.DataFrame:
    df = pd.read_csv(FULL_DATA_PATH)
    required = {"ID", *ELEMENTS, "HV"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Full dataset is missing columns: {sorted(missing)}")
    return df


@st.cache_data
def load_candidates() -> pd.DataFrame:
    df = pd.read_csv(CANDIDATE_PATH)
    required = {"Candidate", *ELEMENTS}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Candidate dataset is missing columns: {sorted(missing)}")
    return df.sort_values("Candidate").reset_index(drop=True)


@st.cache_resource
def load_model() -> ExtraTreesRegressor:
    """Rebuild the exact classroom model from the 140 training samples.

    This avoids cloud deployment problems caused by loading a serialized
    scikit-learn model produced under a different Python/package version.
    """
    full = load_full_data().copy()
    class_df = load_candidates().copy()

    if "ID" in class_df.columns:
        holdout_ids = class_df["ID"].astype(int).tolist()
    else:
        holdout_ids = FALLBACK_HOLDOUT_IDS

    train = full.loc[~full["ID"].astype(int).isin(holdout_ids)].copy()
    if len(train) != 140:
        raise RuntimeError(
            f"Expected 140 training samples after removing the 15 classroom holdouts; found {len(train)}."
        )

    X = train[ELEMENTS].div(train[ELEMENTS].sum(axis=1), axis=0).to_numpy(dtype=float) * 100.0
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



@st.cache_data
def get_training_data() -> pd.DataFrame:
    """Return the same 140-sample training set used by the classroom model."""
    full = load_full_data().copy()
    class_df = load_candidates().copy()
    if "ID" in class_df.columns:
        holdout_ids = class_df["ID"].astype(int).tolist()
    else:
        holdout_ids = FALLBACK_HOLDOUT_IDS
    train = full.loc[~full["ID"].astype(int).isin(holdout_ids)].copy()
    return train.reset_index(drop=True)


def training_matrix() -> np.ndarray:
    train = get_training_data()
    return train[ELEMENTS].div(train[ELEMENTS].sum(axis=1), axis=0).to_numpy(dtype=float) * 100.0


@st.cache_data
def training_domain_summary() -> pd.DataFrame:
    X = training_matrix()
    rows = []
    for j, e in enumerate(ELEMENTS):
        vals = X[:, j]
        rows.append(
            {
                "Element": e,
                "Min": np.min(vals),
                "5th percentile": np.quantile(vals, 0.05),
                "Median": np.median(vals),
                "95th percentile": np.quantile(vals, 0.95),
                "Max": np.max(vals),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def training_nearest_neighbour_distances() -> np.ndarray:
    """Nearest-neighbour Euclidean distance in the six-dimensional at.% space."""
    X = training_matrix()
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)


def domain_diagnostics(comp: dict[str, float] | pd.Series) -> dict:
    """Simple classroom coverage diagnostic, not a formal uncertainty estimate."""
    q = normalize_for_model(comp)
    X = training_matrix()
    summary = training_domain_summary().set_index("Element")

    q_to_train = np.sqrt(((X - q[None, :]) ** 2).sum(axis=1))
    nearest = float(np.min(q_to_train))
    nn_train = training_nearest_neighbour_distances()
    p90 = float(np.quantile(nn_train, 0.90))
    p95 = float(np.quantile(nn_train, 0.95))

    outside_minmax = []
    outside_central = []
    rows = []
    for i, e in enumerate(ELEMENTS):
        lo = float(summary.loc[e, "Min"])
        p5 = float(summary.loc[e, "5th percentile"])
        med = float(summary.loc[e, "Median"])
        p95e = float(summary.loc[e, "95th percentile"])
        hi = float(summary.loc[e, "Max"])
        val = float(q[i])
        if val < lo - 1e-9 or val > hi + 1e-9:
            outside_minmax.append(e)
        if val < p5 - 1e-9 or val > p95e + 1e-9:
            outside_central.append(e)
        rows.append(
            {
                "Element": e,
                "Your alloy (at.%)": val,
                "Training min": lo,
                "Training 5%": p5,
                "Training median": med,
                "Training 95%": p95e,
                "Training max": hi,
            }
        )

    if outside_minmax:
        label = "OUTSIDE observed marginal range"
        level = "high"
        explanation = (
            "At least one elemental fraction lies outside the values observed in the 140 training alloys. "
            "The model will still return a number, but that number should be treated as extrapolative and low-confidence."
        )
    elif nearest > p95:
        label = "SPARSE / compositionally unusual"
        level = "high"
        explanation = (
            "Each elemental fraction may look individually plausible, but the six-dimensional combination is unusually far from the nearest training alloy."
        )
    elif outside_central or nearest > p90:
        label = "EDGE of training domain"
        level = "medium"
        explanation = (
            "The composition is within the observed training envelope, but it lies near the edge of the data or in a relatively sparse region."
        )
    else:
        label = "WELL COVERED by training data"
        level = "low"
        explanation = (
            "The composition lies within the central elemental ranges and is not unusually far from the nearest training composition. "
            "This does not guarantee correctness, but it is more interpolation-like than extrapolative."
        )

    return {
        "label": label,
        "level": level,
        "explanation": explanation,
        "nearest_distance": nearest,
        "train_nn_p90": p90,
        "train_nn_p95": p95,
        "outside_minmax": outside_minmax,
        "outside_central": outside_central,
        "table": pd.DataFrame(rows),
        "query": q,
    }


def nearest_training_examples(comp: dict[str, float] | pd.Series, k: int = 5) -> pd.DataFrame:
    """Return the k compositionally nearest training alloys in six-dimensional at.% space."""
    q = normalize_for_model(comp)
    train = get_training_data().copy()
    X = train[ELEMENTS].div(train[ELEMENTS].sum(axis=1), axis=0).to_numpy(dtype=float) * 100.0
    distances = np.sqrt(((X - q[None, :]) ** 2).sum(axis=1))
    idx = np.argsort(distances)[:k]

    out = train.iloc[idx][["ID"] + ELEMENTS + ["HV"]].copy()
    out["Distance"] = distances[idx]
    out = out.rename(columns={"HV": "Experimental HV"})
    return out.reset_index(drop=True)



def _mini_hist_html(values: np.ndarray, bins: int = 12) -> str:
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 100.0))
    max_count = max(int(counts.max()), 1)
    bars = []
    for c in counts:
        h = 8 + int(50 * float(c) / max_count)
        bars.append(f'<span class="mini-bar" style="height:{h}px" title="{int(c)} training alloys"></span>')
    return ''.join(bars)


def training_distribution_card(element: str) -> None:
    idx = ELEMENTS.index(element)
    vals = training_matrix()[:, idx]
    s = training_domain_summary().set_index("Element").loc[element]
    bars = _mini_hist_html(vals, bins=12)
    st.markdown(
        f'''
        <div class="dist-card">
          <div class="dist-head">
            <span class="dist-element">{element}</span>
            <span class="dist-value">median {float(s['Median']):.1f} at.%</span>
          </div>
          <div class="mini-hist">{bars}</div>
          <div class="range-axis"><span>0</span><span>25</span><span>50</span><span>75</span><span>100 at.%</span></div>
          <div class="dist-stats">
            Typical middle 90%: <strong>{float(s['5th percentile']):.1f}–{float(s['95th percentile']):.1f} at.%</strong><br>
            Observed training range: {float(s['Min']):.1f}–{float(s['Max']):.1f} at.%
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def element_coverage_card(element: str, query_value: float) -> None:
    s = training_domain_summary().set_index("Element").loc[element]
    lo = float(s["Min"]); p5 = float(s["5th percentile"]); med = float(s["Median"])
    p95 = float(s["95th percentile"]); hi = float(s["Max"]); val = float(query_value)

    lo_p = max(0.0, min(100.0, lo)); hi_p = max(0.0, min(100.0, hi))
    p5_p = max(0.0, min(100.0, p5)); p95_p = max(0.0, min(100.0, p95))
    med_p = max(0.0, min(100.0, med)); val_p = max(0.0, min(100.0, val))

    if val < lo - 1e-9 or val > hi + 1e-9:
        label, cls = "outside observed range", "status-high"
    elif val < p5 - 1e-9 or val > p95 + 1e-9:
        label, cls = "edge of distribution", "status-medium"
    else:
        label, cls = "within typical range", "status-low"

    st.markdown(
        f'''
        <div class="range-card">
          <div class="range-head">
            <span class="range-element">{element}</span>
            <span class="range-query">Your alloy: {val:.1f} at.%</span>
          </div>
          <div><span class="status-pill {cls}">{label}</span></div>
          <div class="range-track">
            <span class="range-observed" style="left:{lo_p:.2f}%; width:{max(0.4, hi_p-lo_p):.2f}%"></span>
            <span class="range-typical" style="left:{p5_p:.2f}%; width:{max(0.4, p95_p-p5_p):.2f}%"></span>
            <span class="range-median" style="left:{med_p:.2f}%" title="Training median"></span>
            <span class="range-query-marker" style="left:{val_p:.2f}%" title="Your alloy"></span>
          </div>
          <div class="range-axis"><span>0</span><span>25</span><span>50</span><span>75</span><span>100 at.%</span></div>
          <div class="range-note">
            Typical (5–95%): <strong>{p5:.1f}–{p95:.1f}</strong> &nbsp;·&nbsp;
            median: {med:.1f} &nbsp;·&nbsp; observed: {lo:.1f}–{hi:.1f}
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def coverage_status_panel(diagnostic: dict) -> None:
    level = diagnostic["level"]
    pill_cls = {"low": "status-low", "medium": "status-medium", "high": "status-high"}.get(level, "status-medium")
    concern = {"low": "Lower coverage concern", "medium": "Moderate coverage concern", "high": "High coverage concern"}.get(level, "Coverage check")

    table = diagnostic["table"]
    in_observed = int(((table["Your alloy (at.%)"] >= table["Training min"]) & (table["Your alloy (at.%)"] <= table["Training max"])).sum())
    in_typical = int(((table["Your alloy (at.%)"] >= table["Training 5%"]) & (table["Your alloy (at.%)"] <= table["Training 95%"])) .sum())

    nearest = float(diagnostic["nearest_distance"])
    p90 = float(diagnostic["train_nn_p90"])
    p95 = float(diagnostic["train_nn_p95"])
    axis_max = max(p95 * 1.45, nearest * 1.15, 1.0)
    p90_pos = min(100.0, 100.0 * p90 / axis_max)
    p95_pos = min(100.0, 100.0 * p95 / axis_max)
    q_pos = min(100.0, 100.0 * nearest / axis_max)

    st.markdown(
        f'''
        <div class="coverage-panel">
          <div class="coverage-title">Training-domain coverage</div>
          <span class="status-pill {pill_cls}">{concern}</span>
          <div class="coverage-big">{diagnostic['label']}</div>
          <div class="coverage-explain">{diagnostic['explanation']}</div>
          <div style="margin-top:.75rem; font-size:.86rem; color:#334155;">
            <strong>Element-wise:</strong> {in_typical}/6 elements lie inside the central 5–95% training band; {in_observed}/6 lie inside the observed min–max range.
          </div>
          <div style="margin-top:.55rem; font-size:.86rem; color:#334155;">
            <strong>Whole-composition:</strong> nearest training-alloy distance = {nearest:.1f}. Compare this with the training-set 90th/95th-percentile neighbour distances ({p90:.1f}/{p95:.1f}).
          </div>
          <div class="distance-track">
            <span class="distance-safe" style="width:{p90_pos:.2f}%"></span>
            <span class="distance-edge" style="left:{p90_pos:.2f}%; width:{max(0.0,p95_pos-p90_pos):.2f}%"></span>
            <span class="distance-marker" style="left:{q_pos:.2f}%" title="Your alloy"></span>
          </div>
          <div class="distance-labels"><span>closer to training chemistry</span><span>more unusual →</span></div>
          <div class="legend-line">
            <span class="legend-key"><span class="legend-swatch" style="background:#DCEFE1"></span>usual training neighbourhood</span>
            <span class="legend-key"><span class="legend-swatch" style="background:#FBE7B7"></span>edge region</span>
            <span class="legend-key"><span style="display:inline-block;width:4px;height:14px;background:#1F2937;border-radius:2px"></span>your alloy</span>
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.caption(
        "This is a training-coverage diagnostic, not a calibrated uncertainty interval. A well-covered composition can still be predicted incorrectly, and a sparse composition can still happen to be predicted well."
    )


def pca_domain_plot(comp: dict[str, float] | pd.Series) -> None:
    X = training_matrix()
    q = normalize_for_model(comp)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    qz = pca.transform(q.reshape(1, -1))[0]
    explained = 100 * (pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])

    st.markdown(
        """
        <div class="concept-card">
        <b>What is PCA, in plain language?</b><br><br>
        Each alloy is described by <b>six composition numbers</b>: Al, Co, Cr, Cu, Fe and Ni.
        That means every alloy really lives in a six-dimensional composition space, which we cannot draw directly.
        <br><br>
        <b>Principal Component Analysis (PCA)</b> compresses those six composition variables into two new axes,
        called <b>PC1</b> and <b>PC2</b>, chosen to preserve as much of the composition variation as possible.
        You can think of it as taking the six-dimensional cloud of alloys and viewing it from a useful angle so that it fits on a 2-D page.
        <br><br>
        <b>Important:</b> PC1 and PC2 are mathematical combinations of Al, Co, Cr, Cu, Fe and Ni.
        They are <b>not physical properties</b>, and neither axis means "hardness".
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**How should you read the PCA map?**")
    st.markdown(
        """
        - Each small point represents one alloy used to train the model.
        - The large point represents **your entered alloy**.
        - If your point sits **inside a dense region of training points**, its composition is similar to alloys the model has seen before.
        - If it sits **far away or isolated**, the composition is more unusual, so the prediction deserves extra caution.
        - Distance on this plot means **similarity in composition patterns**, not difference in hardness.
        """
    )

    plot_df = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1]})
    plot_df["Point"] = "Training alloy"
    plot_df["Size"] = 35
    query_df = pd.DataFrame(
        {"PC1": [qz[0]], "PC2": [qz[1]], "Point": ["Your alloy"], "Size": [250]}
    )
    plot_df = pd.concat([plot_df, query_df], ignore_index=True)
    st.scatter_chart(plot_df, x="PC1", y="PC2", color="Point", size="Size", use_container_width=True)

    st.info(
        f"In this dataset, PC1 and PC2 together retain about {explained:.1f}% of the total variation in the six composition variables. "
        "The remaining variation is not shown in this 2-D picture. Therefore, PCA is a visual aid for judging training-domain coverage, not a formal proof that a composition is in-distribution or out-of-distribution. "
        "That is why the app also checks the full six-dimensional nearest-neighbour distance."
    )

# ============================================================
# CALCULATION HELPERS
# ============================================================


def atomic_fractions(comp: dict[str, float] | pd.Series) -> dict[str, float]:
    values = {e: float(comp[e]) for e in ELEMENTS}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Composition total must be positive.")
    return {e: values[e] / total for e in ELEMENTS}


def normalize_for_model(comp: dict[str, float] | pd.Series) -> np.ndarray:
    vals = np.array([float(comp[e]) for e in ELEMENTS], dtype=float)
    total = vals.sum()
    if total <= 0:
        raise ValueError("Composition total must be positive.")
    return vals / total * 100.0


def descriptors(comp: dict[str, float] | pd.Series) -> dict[str, float]:
    x = atomic_fractions(comp)
    vec = sum(x[e] * VEC[e] for e in ELEMENTS)
    rbar = sum(x[e] * RADIUS_PM[e] for e in ELEMENTS)
    delta = 100.0 * math.sqrt(
        sum(x[e] * (1.0 - RADIUS_PM[e] / rbar) ** 2 for e in ELEMENTS)
    )

    numerator = sum(x[e] * ATOMIC_MASS[e] for e in ELEMENTS)
    denominator = sum(x[e] * ATOMIC_MASS[e] / ELEMENT_DENSITY[e] for e in ELEMENTS)
    rho = numerator / denominator

    mass_parts = {e: x[e] * ATOMIC_MASS[e] for e in ELEMENTS}
    mass_total = sum(mass_parts.values())
    mass_fraction = {e: mass_parts[e] / mass_total for e in ELEMENTS}
    raw_cost = sum(mass_fraction[e] * COST_WEIGHT[e] for e in ELEMENTS)

    return {"VEC": vec, "rbar": rbar, "delta": delta, "density": rho, "raw_cost": raw_cost}


def predict_hv(comp: dict[str, float] | pd.Series) -> float:
    model = load_model()
    X = normalize_for_model(comp).reshape(1, -1)
    return float(model.predict(X)[0])


def prediction_table(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["ML HV"] = [predict_hv(row) for _, row in out.iterrows()]
    out["ML decision"] = np.where(out["ML HV"] >= HARDNESS_REQUIREMENT, "PASS", "Fail")
    out["Margin to 450"] = out["ML HV"] - HARDNESS_REQUIREMENT
    return out


def controlled_five_table(full: pd.DataFrame, reveal_hv: bool) -> pd.DataFrame:
    configs = [
        ("Al", {"Al": 0, "Co": 20, "Cr": 20, "Cu": 20, "Fe": 20, "Ni": 20}),
        ("Fe", {"Al": 20, "Co": 20, "Cr": 20, "Cu": 20, "Fe": 0, "Ni": 20}),
        ("Co", {"Al": 20, "Co": 0, "Cr": 20, "Cu": 20, "Fe": 20, "Ni": 20}),
        ("Cu", {"Al": 20, "Co": 20, "Cr": 20, "Cu": 0, "Fe": 20, "Ni": 20}),
        ("Cr", {"Al": 20, "Co": 20, "Cr": 0, "Cu": 20, "Fe": 20, "Ni": 20}),
    ]
    rows = []
    for omitted, comp in configs:
        d = descriptors(comp)
        mask = np.ones(len(full), dtype=bool)
        for e in ELEMENTS:
            mask &= np.isclose(full[e].astype(float).to_numpy(), comp[e], atol=0.25)
        match = full.loc[mask]
        hv = float(match.iloc[0]["HV"]) if len(match) else np.nan
        label = " ".join(f"{e}{int(comp[e])}" for e in ELEMENTS if comp[e] > 0)
        row = {
            "Alloy": f"omit {omitted}",
            "Composition": label,
            "VEC": d["VEC"],
            "δ (%)": d["delta"],
        }
        if reveal_hv:
            row["Experimental HV"] = hv
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_costs(df: pd.DataFrame) -> pd.Series:
    raw = pd.Series([descriptors(row)["raw_cost"] for _, row in df.iterrows()], index=df.index)
    return raw / raw.min()


def add_selection_metrics(df: pd.DataFrame, all_candidates: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "rho_mix_g_cm3" in out.columns:
        out["Density"] = out["rho_mix_g_cm3"].astype(float)
    else:
        out["Density"] = [descriptors(row)["density"] for _, row in out.iterrows()]
    out["Relative cost"] = candidate_costs(all_candidates).loc[out.index].to_numpy()
    out["Specific hardness"] = out["HV"] / out["Density"]
    out["Hardness margin"] = out["HV"] - HARDNESS_REQUIREMENT
    return out


def dominated_by(df: pd.DataFrame, candidate: str) -> str | None:
    """Dominance on hardness (higher), density (lower), and cost (lower)."""
    row = df.loc[df["Candidate"] == candidate].iloc[0]
    for _, other in df.iterrows():
        if other["Candidate"] == candidate:
            continue
        no_worse = (
            other["HV"] >= row["HV"]
            and other["Density"] <= row["Density"]
            and other["Relative cost"] <= row["Relative cost"]
        )
        strictly_better = (
            other["HV"] > row["HV"]
            or other["Density"] < row["Density"]
            or other["Relative cost"] < row["Relative cost"]
        )
        if no_worse and strictly_better:
            return str(other["Candidate"])
    return None


# ============================================================
# SESSION STATE / NAVIGATION
# ============================================================


def initialise_state() -> None:
    defaults = {
        "stage": 1,
        "s1_complete": False,
        "s2_complete": False,
        "s3_complete": False,
        "s4_complete": False,
        "s5_complete": False,
        "stage2_reveal": True,
        "stage4_reveal": True,
        "model_runs": [],
        "scenario": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def go_to(stage: int) -> None:
    st.session_state.stage = stage
    st.rerun()


def reset_activity() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def progress_header(stage: int) -> None:
    st.progress(stage / len(STAGE_NAMES))
    labels = []
    for i, (name, mins) in enumerate(zip(STAGE_NAMES, STAGE_TIMES), start=1):
        labels.append(f"**{name} ({mins})**" if i == stage else f"{name} ({mins})")
    st.caption("  →  ".join(labels))


def stage_nav(stage: int) -> None:
    st.divider()
    c1, c2, c3 = st.columns([1, 1, 5])
    with c1:
        if stage > 1 and st.button("← Previous", use_container_width=True):
            go_to(stage - 1)
    with c2:
        if stage < len(STAGE_NAMES):
            if st.button("Next →", use_container_width=True):
                go_to(stage + 1)


# ============================================================
# COMMON UI
# ============================================================

initialise_state()
full_df = load_full_data()
candidates = load_candidates()

with st.sidebar:
    st.header("60-minute group practical")
    st.write("**Group size:** 4 students")
    st.write("**Design threshold:** 450 HV")
    st.caption("450 HV is a teaching requirement for this exercise, not a universal industry standard.")
    st.divider()
    st.caption("Supervisor preview: all stages are freely accessible.")
    for i, (name, mins) in enumerate(zip(STAGE_NAMES, STAGE_TIMES), start=1):
        if st.button(
            f"{'➡️ ' if st.session_state.stage == i else ''}{name} — {mins}",
            key=f"preview_stage_{i}",
            use_container_width=True,
        ):
            go_to(i)
    st.divider()
    if st.button("Restart practical", use_container_width=True):
        reset_activity()
    with st.expander("Model scope"):
        st.write(
            "The model is a classroom model for the as-cast Al–Co–Cr–Cu–Fe–Ni composition family represented by the dataset. "
            "It should not be treated as a universal hardness model or used for large extrapolations."
        )

st.title("HEA hardness: physics → ML → experiment → materials selection")
st.info(
    "SUPERVISOR PREVIEW MODE — all stages can be opened directly and Next is always available. "
    "Student prompts remain visible, but you do not need to complete them to inspect the app."
)
st.write(
    "This preview shows the same teaching content as the student build, including the physical reasoning, "
    "interactive ML/OOD investigation, experimental error-detective activity, and final materials-selection challenge."
)

with st.expander("Team roles — read before starting", expanded=(st.session_state.stage == 1)):
    st.markdown(
        """
Assign one role to each student. Rotate who speaks for the group at each stage.

- **Student A — Calculation lead:** checks equations, units and arithmetic.
- **Student B — Materials-mechanism lead:** explains why a descriptor might influence deformation/hardness.
- **Student C — Evidence/research lead:** looks for an additional useful material descriptor or microstructural factor.
- **Student D — Critical reviewer:** challenges assumptions, asks what information is missing, and watches for unsafe engineering conclusions.

In Stage 5, switch roles to **performance advocate, cost advocate, lightweight-design advocate, and decision chair**.
"""
    )

progress_header(st.session_state.stage)

# ============================================================
# STAGE 1 — PHYSICAL DESCRIPTORS
# ============================================================

if st.session_state.stage == 1:
    st.markdown('<div class="stage-title">Stage 1 — Build a physical expectation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aim-box"><strong>Aim:</strong> calculate physically meaningful composition descriptors for two real alloys, then use them to form a hardness hypothesis before seeing any hardness value.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("A. Why these descriptors might matter")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("### Valence electron concentration (VEC)")
        st.latex(r"VEC=\sum_i x_iVEC_i")
        st.write(
            "VEC is the composition-weighted average valence-electron count. In transition-metal-rich multicomponent alloys, "
            "electronic structure is associated with bonding and with tendencies toward different crystal structures/phases. "
            "Because phase constitution and bonding influence slip and plastic deformation, VEC can carry useful information about hardness."
        )
        st.warning("VEC is **not** a hardness equation. A lower or higher VEC does not guarantee a particular hardness for every alloy.")

    with p2:
        st.markdown("### Atomic-size mismatch (δ)")
        st.latex(r"\bar r=\sum_i x_ir_i")
        st.latex(r"\delta=100\sqrt{\sum_i x_i\left(1-\frac{r_i}{\bar r}\right)^2}")
        st.write(
            "Atoms of different sizes disturb the local lattice. Those local strain fields can interact with dislocations, "
            "so greater size mismatch can contribute to solid-solution strengthening and resistance to plastic deformation."
        )
        st.warning("δ is also **not** a hardness equation. Phase constitution, microstructure, processing and other effects can change the outcome.")

    st.info(
        "The mean atomic radius r̄ is mainly an intermediate quantity required to calculate δ. Do not interpret r̄ alone as a direct hardness predictor."
    )

    constants = pd.DataFrame(
        {
            "Element": ELEMENTS,
            "VEC value": [VEC[e] for e in ELEMENTS],
            "Atomic radius (pm)": [RADIUS_PM[e] for e in ELEMENTS],
        }
    )
    st.dataframe(constants, hide_index=True, use_container_width=True)
    st.caption("Remember: 20 at.% = atomic fraction 0.20 in the equations.")

    alloy_c = {"Al": 0, "Co": 20, "Cr": 20, "Cu": 20, "Fe": 20, "Ni": 20}
    alloy_k = {"Al": 20, "Co": 20, "Cr": 20, "Cu": 0, "Fe": 20, "Ni": 20}

    st.subheader("B. Calculate two real samples")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Alloy C — Al omitted**")
        st.code("Co20 Cr20 Cu20 Fe20 Ni20", language=None)
        c_vec = st.number_input("Alloy C — VEC", 0.0, 12.0, step=0.01, key="s1_c_vec")
        c_r = st.number_input("Alloy C — mean radius r̄ (pm)", 100.0, 160.0, step=0.01, key="s1_c_r")
        c_delta = st.number_input("Alloy C — δ (%)", 0.0, 20.0, step=0.01, key="s1_c_delta")
    with c2:
        st.markdown("**Alloy K — Cu omitted**")
        st.code("Al20 Co20 Cr20 Fe20 Ni20", language=None)
        k_vec = st.number_input("Alloy K — VEC", 0.0, 12.0, step=0.01, key="s1_k_vec")
        k_r = st.number_input("Alloy K — mean radius r̄ (pm)", 100.0, 160.0, step=0.01, key="s1_k_r")
        k_delta = st.number_input("Alloy K — δ (%)", 0.0, 20.0, step=0.01, key="s1_k_delta")

    expectation = st.radio(
        "Before seeing hardness: which alloy do you expect to resist local plastic deformation more strongly?",
        ["Choose one…", "Alloy C", "Alloy K", "I cannot justify a difference from these descriptors"],
        key="s1_expectation",
    )
    reasoning = st.text_area(
        "Give the group's physical reason in 1–3 sentences.",
        placeholder="Use VEC and/or atomic-size mismatch. Do not claim that either descriptor determines hardness exactly.",
        key="s1_reason",
        height=90,
    )

    if st.button("Check Stage 1", type="primary"):
        calc_ok = (
            abs(c_vec - 8.80) <= 0.05
            and abs(c_r - 126.20) <= 0.10
            and abs(c_delta - 1.27) <= 0.08
            and abs(k_vec - 7.20) <= 0.05
            and abs(k_r - 129.20) <= 0.10
            and abs(k_delta - 5.44) <= 0.08
        )
        if expectation == "Choose one…":
            st.error("Commit to a physical expectation before checking.")
        elif len(reasoning.strip()) < 30:
            st.error("Add a brief physical justification. The reasoning matters as much as the arithmetic.")
        elif not calc_ok:
            st.error("At least one calculation is outside the accepted tolerance. Re-check the weighted sums and the 20 at.% → 0.20 conversion.")
        else:
            st.session_state.s1_complete = True
            st.success("Calculations accepted.")
            st.markdown(
                '<div class="takeaway"><strong>What changed when Cu was replaced by Al?</strong><br>'
                'VEC: <strong>8.80 → 7.20</strong><br>'
                'Mean radius: <strong>126.20 → 129.20 pm</strong><br>'
                'δ: <strong>1.27% → 5.44%</strong><br><br>'
                'You now have a physical hypothesis. Stage 2 asks whether that hypothesis survives a broader controlled comparison.</div>',
                unsafe_allow_html=True,
            )

    stage_nav(1)

# ============================================================
# STAGE 2 — RESEARCH + HYPOTHESIS
# ============================================================

elif st.session_state.stage == 2:
    st.markdown('<div class="stage-title">Stage 2 — Research, extend the physics, then rank five real alloys</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aim-box"><strong>Aim:</strong> recognise that VEC and δ are only part of the story. Find other plausible hardness-controlling variables, then use all your reasoning to make a testable ranking before experiment is revealed.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="research-box"><strong>5-minute research task:</strong> use your lecture notes, textbook, or a quick literature/web search. Find <strong>two additional properties, descriptors or microstructural factors</strong> that could plausibly help explain or rank hardness in multicomponent alloys. Do not simply search for the answer to this dataset.</div>',
        unsafe_allow_html=True,
    )

    st.write("For each idea, record **what it is**, **why it could affect hardness**, and whether it can be obtained from composition alone.")

    r1, r2 = st.columns(2)
    with r1:
        prop1 = st.text_input("Property / factor 1", key="s2_prop1")
        mech1 = st.text_area("Why could it influence hardness?", key="s2_mech1", height=90)
        comp1 = st.radio("Can it be obtained from composition alone?", ["Choose…", "Yes", "No / not uniquely"], key="s2_comp1")
    with r2:
        prop2 = st.text_input("Property / factor 2", key="s2_prop2")
        mech2 = st.text_area("Why could it influence hardness?", key="s2_mech2", height=90)
        comp2 = st.radio("Can it be obtained from composition alone?", ["Choose…", "Yes", "No / not uniquely"], key="s2_comp2")

    source_note = st.text_input(
        "Where did you look? Give one short source note (e.g. textbook chapter, lecture notes, paper title, or search terms).",
        key="s2_source",
    )

    with st.expander("Only open after your group has searched: examples of reasonable directions"):
        st.write(
            "Possible ideas include mixing enthalpy, electronegativity mismatch, elastic/shear modulus, phase constitution, grain size, precipitates, "
            "degree of ordering, or processing-induced microstructure. Some can be approximated from elemental data; others require structural or experimental information."
        )
        st.info(
            "This distinction matters: if an important variable cannot be inferred uniquely from composition, a composition-only ML model cannot directly observe it either."
        )

    st.subheader("Now rank five real controlled alloys")
    st.write(
        "Each alloy contains five elements at 20 at.%; one element is omitted. The table gives only the simple descriptors you already know. "
        "Use VEC, δ, and the additional ideas your group researched to predict the hardness order."
    )

    five_hidden = controlled_five_table(full_df, reveal_hv=False)
    st.dataframe(
        five_hidden.style.format({"VEC": "{:.2f}", "δ (%)": "{:.2f}"}),
        hide_index=True,
        use_container_width=True,
    )

    labels = five_hidden["Alloy"].tolist()
    ranking = st.multiselect(
        "Select ALL FIVE alloys in your predicted order from SOFTEST → HARDEST.",
        labels,
        max_selections=5,
        key="s2_ranking",
    )
    confidence = st.radio("Confidence in your ranking", ["Low", "Medium", "High"], horizontal=True, key="s2_conf")
    main_factor = st.text_input("Which physical idea influenced your ranking most?", key="s2_mainfactor")

    research_complete = (
        len(prop1.strip()) >= 2
        and len(mech1.strip()) >= 20
        and comp1 != "Choose…"
        and len(prop2.strip()) >= 2
        and len(mech2.strip()) >= 20
        and comp2 != "Choose…"
        and len(source_note.strip()) >= 3
    )

    if not st.session_state.stage2_reveal:
        if st.button("Lock our prediction and reveal experiment", type="primary"):
            if not research_complete:
                st.error("Complete the two short research entries before revealing the experiment.")
            elif len(ranking) != 5 or len(set(ranking)) != 5:
                st.error("Enter a complete five-alloy ranking from softest to hardest.")
            elif len(main_factor.strip()) < 3:
                st.error("State which idea influenced your ranking most.")
            else:
                st.session_state.stage2_reveal = True
                st.session_state.s2_locked_ranking = list(ranking)
                st.rerun()

    if st.session_state.stage2_reveal:
        st.subheader("Experimental reveal")
        five_real = controlled_five_table(full_df, reveal_hv=True).sort_values("Experimental HV")
        st.dataframe(
            five_real.style.format({"VEC": "{:.2f}", "δ (%)": "{:.2f}", "Experimental HV": "{:.0f}"}),
            hide_index=True,
            use_container_width=True,
        )

        reveal_chart = five_real[["Alloy", "Experimental HV"]].to_dict(orient="records")
        st.vega_lite_chart(
            {
                "height": 280,
                "data": {"values": reveal_chart},
                "mark": {"type": "bar"},
                "encoding": {
                    "x": {"field": "Alloy", "type": "nominal", "sort": {"field": "Experimental HV"}, "title": "Controlled alloy"},
                    "y": {"field": "Experimental HV", "type": "quantitative", "title": "Measured hardness (HV)"},
                    "tooltip": [
                        {"field": "Alloy", "type": "nominal"},
                        {"field": "Experimental HV", "type": "quantitative", "format": ".0f"},
                    ],
                },
            },
            use_container_width=True,
        )

        actual_order = " → ".join(five_real["Alloy"].tolist())
        predicted_order = " → ".join(st.session_state.get("s2_locked_ranking", []))
        st.write(f"**Your prediction:** {predicted_order}")
        st.write(f"**Measured softest → hardest:** {actual_order}")

        st.markdown(
            """
**Two important observations**

1. The five alloys are all equiatomic five-component alloys, so their ideal configurational entropy is the same, yet hardness spans roughly **155–536 HV**. Entropy alone therefore cannot rank them.
2. VEC and δ show useful broad trends, but neither gives a unique mapping to hardness. For example, two alloys can have nearly the same δ and still differ in hardness.
"""
        )

        reflection = st.text_area(
            "Group reflection: what did your ranking get right, what did it miss, and what information might be missing from VEC and δ?",
            key="s2_reflection",
            height=120,
        )

        if st.button("Complete Stage 2"):
            if len(reflection.strip()) < 80:
                st.error("Give a fuller reflection — aim for at least 2–4 sentences.")
            else:
                st.session_state.s2_complete = True
                st.success("Stage 2 complete.")
                st.markdown(
                    '<div class="takeaway"><strong>Core intuition:</strong> physically motivated descriptors are valuable for interpretation, but a complex alloy property is multivariable. Some important variables may not even be uniquely determined by composition. This is the reason to test a data-driven model next — not because physics is irrelevant, but because the simple descriptors are incomplete.</div>',
                    unsafe_allow_html=True,
                )

    stage_nav(2)

# ============================================================
# STAGE 3 — INTERACTIVE ML LAB + TRAINING DOMAIN
# ============================================================

elif st.session_state.stage == 3:
    st.markdown('<div class="stage-title">Stage 3 — Interactive ML laboratory: prediction and training-domain awareness</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aim-box"><strong>Aim:</strong> inspect what data the model actually learned from, use the model yourselves, and decide whether a numerical prediction is interpolation-like or potentially out-of-distribution before trusting it.</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "The Extra Trees model is rebuilt from **140 experimental compositions**. The 15 classroom candidates are excluded from training. "
        "Its only inputs are the six elemental atomic percentages — it does not directly see phase fraction, grain size, defects, heat treatment or other microstructural variables."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("10-fold CV R²", "0.919")
    m2.metric("10-fold CV MAE", "36.9 HV")
    m3.metric("Training samples", "140")
    st.caption("These are average predictive-performance measures. They do not tell you whether a particular new composition is well represented by the training data.")

    st.subheader("A. First inspect the model's training domain")
    st.write(
        "Before asking a model for a prediction, ask a more basic question: **has it seen chemistry like this before?** "
        "Use the plots below to understand the elemental ranges and density of the 140 training compositions."
    )

    dist_tab, range_tab, hv_tab = st.tabs(["Element distributions", "Numerical coverage", "Training hardness distribution"])

    with dist_tab:
        rows = [st.columns(3), st.columns(3)]
        for idx, e in enumerate(ELEMENTS):
            with rows[idx // 3][idx % 3]:
                training_distribution_card(e)

    with range_tab:
        domain_summary = training_domain_summary()
        st.dataframe(
            domain_summary.style.format({
                "Min": "{:.1f}", "5th percentile": "{:.1f}", "Median": "{:.1f}",
                "95th percentile": "{:.1f}", "Max": "{:.1f}"
            }),
            hide_index=True,
            use_container_width=True,
        )
        st.info(
            "The min–max range is only a first check. A composition can sit inside every individual elemental range and still be unusual as a **combination** of six elements."
        )

    with hv_tab:
        hv_train = get_training_data()["HV"].astype(float)
        spec = {
            "height": 260,
            "data": {"values": [{"HV": float(v)} for v in hv_train]},
            "mark": {"type": "bar", "opacity": 0.75},
            "encoding": {
                "x": {"field": "HV", "type": "quantitative", "bin": {"maxbins": 18}, "title": "Experimental hardness in training set (HV)"},
                "y": {"aggregate": "count", "type": "quantitative", "title": "Training alloys"},
                "tooltip": [{"aggregate": "count", "type": "quantitative", "title": "Number of alloys"}],
            },
        }
        st.vega_lite_chart(spec, use_container_width=True)
        st.caption(
            f"Training hardness spans approximately {hv_train.min():.0f}–{hv_train.max():.0f} HV. "
            "Predictions in poorly represented chemistry or property regions deserve extra caution."
        )

    domain_observation = st.text_area(
        "Group observation: write one thing you notice about the training-data coverage. Which element(s) seem especially broad, narrow, sparse or concentrated?",
        key="s3_domain_observation",
        height=90,
        placeholder="Example structure: Cu is often absent or relatively low, while ... This means a composition with ... may be less well represented.",
    )

    st.subheader("B. Use the model yourselves")
    st.write(
        "Enter an Al–Co–Cr–Cu–Fe–Ni composition. The app normalizes your numbers to 100 at.% before prediction. "
        "Each time you run the model, the app also checks how well that composition is covered by the 140 training alloys."
    )

    preset = st.selectbox(
        "Start from a preset or build your own",
        ["Alloy C", "Alloy K", "Candidate H", "Candidate I", "Deliberately unusual alloy", "Custom composition"],
        key="s3_preset",
    )

    preset_map = {
        "Alloy C": {"Al": 0, "Co": 20, "Cr": 20, "Cu": 20, "Fe": 20, "Ni": 20},
        "Alloy K": {"Al": 20, "Co": 20, "Cr": 20, "Cu": 0, "Fe": 20, "Ni": 20},
        "Deliberately unusual alloy": {"Al": 70, "Co": 6, "Cr": 6, "Cu": 6, "Fe": 6, "Ni": 6},
    }
    for cand in ["H", "I"]:
        row = candidates.loc[candidates["Candidate"] == cand]
        if len(row):
            preset_map[f"Candidate {cand}"] = {e: float(row.iloc[0][e]) for e in ELEMENTS}
    preset_map["Custom composition"] = {e: (100.0 / 6.0) for e in ELEMENTS}

    base = preset_map[preset]
    cols = st.columns(6)
    entered = {}
    for col, e in zip(cols, ELEMENTS):
        with col:
            entered[e] = st.number_input(
                e,
                min_value=0.0,
                max_value=100.0,
                value=float(base[e]),
                step=0.5,
                key=f"s3_{preset}_{e}",
            )

    total = sum(entered.values())
    st.caption(f"Entered total = {total:.1f} at.%. The app normalizes the composition to 100 at.% before modelling.")

    diagnostic = None
    if total > 0:
        diagnostic = domain_diagnostics(entered)
        norm = diagnostic["query"]

        st.markdown("### Training-domain check for your alloy")
        coverage_status_panel(diagnostic)

        st.markdown("#### 1. Element-by-element coverage")
        st.caption(
            "The pale bar is the full observed training range, the darker band is the central 5–95% of the training data, the thin dark line is the training median, and the red marker is your alloy."
        )
        mini_cols = [st.columns(3), st.columns(3)]
        for idx, e in enumerate(ELEMENTS):
            with mini_cols[idx // 3][idx % 3]:
                element_coverage_card(e, norm[idx])

        st.markdown("#### 2. Whole-composition view")
        st.write(
            "Element-by-element checks are not enough: a combination can be unusual even when every individual element lies inside its observed range. The PCA view gives a visual projection of the six-dimensional chemistry."
        )
        pca_domain_plot(entered)

        with st.expander("Show exact numerical comparison with the training ranges"):
            st.dataframe(
                diagnostic["table"].style.format({c: "{:.1f}" for c in diagnostic["table"].columns if c != "Element"}),
                hide_index=True,
                use_container_width=True,
            )

    if st.button("Run model prediction", type="primary"):
        if total <= 0:
            st.error("Composition total must be greater than zero.")
        else:
            d = descriptors(entered)
            hv = predict_hv(entered)
            diag = domain_diagnostics(entered)
            run = {
                "Run": len(st.session_state.model_runs) + 1,
                "Composition": ", ".join(f"{e}{normalize_for_model(entered)[i]:.1f}" for i, e in enumerate(ELEMENTS)),
                "VEC": d["VEC"],
                "δ (%)": d["delta"],
                "Predicted HV": hv,
                "Decision": "PASS" if hv >= HARDNESS_REQUIREMENT else "Fail",
                "Training-domain flag": diag["label"],
                "Nearest distance": diag["nearest_distance"],
            }
            st.session_state.model_runs.append(run)
            st.rerun()

    if st.session_state.model_runs:
        runs = pd.DataFrame(st.session_state.model_runs)
        st.dataframe(
            runs.style.format({"VEC": "{:.2f}", "δ (%)": "{:.2f}", "Predicted HV": "{:.0f}", "Nearest distance": "{:.1f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.info(
            "Run at least **three compositions**: (1) one familiar/central composition, (2) one held-out candidate, and (3) one deliberately unusual composition. "
            "Compare not only the predicted HV, but also the training-domain flag."
        )

    ood_question = st.radio(
        "A very unusual composition receives a prediction of 650 HV. What is the best interpretation?",
        [
            "Choose one…",
            "650 HV is trustworthy because the model always returns a numerical value.",
            "The number may be useful as a hypothesis, but confidence should be low if the composition lies outside or in a sparse part of the training domain.",
            "Any out-of-distribution prediction is guaranteed to be wrong.",
        ],
        key="s3_ood_question",
    )

    model_interpretation = st.radio(
        "What does a successful ML prediction mean?",
        [
            "Choose one…",
            "The model has discovered the true physical mechanism of hardness.",
            "The model has learned an empirical composition–hardness relationship that can be useful within its data domain, but this does not prove a causal mechanism.",
        ],
        key="s3_interpret",
    )

    st.subheader("C. Screen the 15 held-out candidates")
    st.write(
        "Now apply the model to all 15 real held-out candidates. The candidates were not used to fit the model. "
        "Use both the predicted hardness and the **margin above the 450-HV requirement** when judging screening risk."
    )

    pred = prediction_table(candidates)
    # Add coverage flags to the screening table.
    flags, distances = [], []
    for _, row in pred.iterrows():
        diag = domain_diagnostics(row)
        flags.append(diag["label"])
        distances.append(diag["nearest_distance"])
    pred["Domain flag"] = flags
    pred["NN distance"] = distances

    display = pred[["Candidate", *ELEMENTS, "ML HV", "ML decision", "Margin to 450", "Domain flag"]].copy()
    st.dataframe(
        display.style.format({**{e: "{:.1f}" for e in ELEMENTS}, "ML HV": "{:.0f}", "Margin to 450": "{:+.0f}"}),
        hide_index=True,
        use_container_width=True,
    )

    # Visual predicted-hardness screening chart.
    chart_df = pred[["Candidate", "ML HV"]].copy()
    chart_df["Threshold"] = HARDNESS_REQUIREMENT
    spec = {
        "height": 300,
        "data": {"values": chart_df.to_dict(orient="records")},
        "layer": [
            {
                "mark": {"type": "bar"},
                "encoding": {
                    "x": {"field": "Candidate", "type": "nominal", "sort": None},
                    "y": {"field": "ML HV", "type": "quantitative", "title": "Predicted hardness (HV)"},
                    "tooltip": [
                        {"field": "Candidate", "type": "nominal"},
                        {"field": "ML HV", "type": "quantitative", "format": ".1f"},
                    ],
                },
            },
            {
                "mark": {"type": "rule", "strokeDash": [6, 4], "strokeWidth": 3},
                "encoding": {"y": {"datum": HARDNESS_REQUIREMENT}},
            },
        ],
    }
    st.vega_lite_chart(spec, use_container_width=True)
    st.caption("The horizontal line is the 450-HV teaching requirement. Candidates barely above the line deserve more caution than candidates far above it.")

    trust = st.radio(
        "Should every predicted PASS be trusted equally?",
        [
            "Choose one…",
            "Yes — once the predicted hardness is above 450 HV, the prediction should be treated as reliable.",
            "No — only predictions close to 450 HV are risky; a prediction far above 450 HV is automatically reliable.",
            "No — any prediction inside the training domain is automatically reliable, regardless of model error or decision margin.",
            "No definitive reliability conclusion can be made from the predicted hardness alone. We must also consider decision margin, training-data coverage, and the model's typical error; even a prediction far above 450 HV can be weakly supported if the composition is out-of-distribution.",
        ],
        key="s3_trust",
    )
    passers = pred.loc[pred["ML HV"] >= HARDNESS_REQUIREMENT, "Candidate"].tolist()
    risky = st.multiselect(
        "Select the TWO most borderline predicted passes.",
        passers,
        max_selections=2,
        key="s3_risky",
    )

    if st.button("Complete Stage 3"):
        run_flags = [r.get("Training-domain flag", "") for r in st.session_state.model_runs]
        has_noncentral = any(flag != "WELL COVERED by training data" for flag in run_flags)
        if len(domain_observation.strip()) < 45:
            st.error("Record one meaningful observation about the training-data coverage first.")
        elif len(st.session_state.model_runs) < 3:
            st.error("Run at least three interactive model predictions first.")
        elif not has_noncentral:
            st.error("Include at least one edge/sparse/outside-domain composition so you can compare it with a well-covered prediction.")
        elif not ood_question.startswith("The number may be useful"):
            st.error("A numerical prediction can still be weakly supported if the input chemistry is outside the model's training domain.")
        elif not model_interpretation.startswith("The model has learned"):
            st.error("Distinguish predictive association from proof of physical mechanism.")
        elif not trust.startswith("No definitive reliability conclusion"):
            st.error(
                "The predicted hardness alone is not enough to establish reliability. Consider decision margin, training-data coverage, and typical model error. "
                "Even a value far above 450 HV can be weakly supported when the chemistry is unfamiliar to the model."
            )
        elif set(risky) != {"H", "I"}:
            st.error("Look for the two predicted passes closest to 450 HV.")
        else:
            st.session_state.s3_complete = True
            st.success("Stage 3 complete.")
            st.markdown(
                '<div class="takeaway"><strong>Core intuition:</strong> the predicted hardness by itself does not tell you how trustworthy the prediction is. Ask: <strong>(1) how far is it from the engineering decision boundary?</strong> <strong>(2) how well is the chemistry represented in training?</strong> and <strong>(3) what size of error does the model normally make?</strong> A high predicted value can still be weakly supported.</div>',
                unsafe_allow_html=True,
            )

    stage_nav(3)

# ============================================================
# STAGE 4 — EXPERIMENTAL VALIDATION + ERROR DETECTIVE
# ============================================================

elif st.session_state.stage == 4:
    st.markdown('<div class="stage-title">Stage 4 — Challenge the model with experiment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aim-box"><strong>Aim:</strong> test whether the ML-based engineering decisions survive real measurements, then diagnose <em>why different prediction errors can arise for different reasons</em>.</div>',
        unsafe_allow_html=True,
    )

    pred = prediction_table(candidates)

    if not st.session_state.stage4_reveal:
        st.write("The experimental hardness values of the 15 held-out candidates are still hidden.")
        st.write("Your group has already identified H and I as risky because their predicted margins above 450 HV are small.")
        if st.button("Reveal the experimental measurements", type="primary"):
            st.session_state.stage4_reveal = True
            st.rerun()

    if st.session_state.stage4_reveal:
        if "HV" not in pred.columns:
            st.error("The candidate CSV must include the instructor `HV` column for Stage 4.")
            st.stop()

        out = pred[["Candidate", "ML HV", "HV"]].copy()
        out["ML decision"] = np.where(out["ML HV"] >= HARDNESS_REQUIREMENT, "PASS", "Fail")
        out["Experimental decision"] = np.where(out["HV"] >= HARDNESS_REQUIREMENT, "PASS", "Fail")
        out["Error (ML − exp.)"] = out["ML HV"] - out["HV"]
        out["Absolute error"] = np.abs(out["Error (ML − exp.)"])
        out = out.rename(columns={"HV": "Experimental HV"})

        st.subheader("1 · What happened when experiment arrived?")
        st.dataframe(
            out.style.format(
                {
                    "ML HV": "{:.0f}",
                    "Experimental HV": "{:.0f}",
                    "Error (ML − exp.)": "{:+.0f}",
                    "Absolute error": "{:.0f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        plot = out.copy()
        plot["Outcome"] = np.select(
            [
                (plot["ML decision"] == "PASS") & (plot["Experimental decision"] == "Fail"),
                (plot["ML decision"] == "PASS") & (plot["Experimental decision"] == "PASS"),
                (plot["ML decision"] == "Fail") & (plot["Experimental decision"] == "Fail"),
            ],
            ["False pass", "Correct pass", "Correct fail"],
            default="Other",
        )
        plot_values = plot[["Candidate", "ML HV", "Experimental HV", "Outcome"]].to_dict(orient="records")

        st.markdown("**Prediction versus experiment**")
        st.vega_lite_chart(
            {
                "height": 360,
                "data": {"values": plot_values},
                "layer": [
                    {
                        "mark": {"type": "point", "filled": True, "size": 115},
                        "encoding": {
                            "x": {"field": "ML HV", "type": "quantitative", "title": "ML prediction (HV)"},
                            "y": {"field": "Experimental HV", "type": "quantitative", "title": "Experimental hardness (HV)"},
                            "color": {"field": "Outcome", "type": "nominal"},
                            "tooltip": [
                                {"field": "Candidate", "type": "nominal"},
                                {"field": "ML HV", "type": "quantitative", "format": ".1f"},
                                {"field": "Experimental HV", "type": "quantitative", "format": ".0f"},
                                {"field": "Outcome", "type": "nominal"},
                            ],
                        },
                    },
                    {
                        "mark": {"type": "text", "dy": -10, "fontSize": 12},
                        "encoding": {
                            "x": {"field": "ML HV", "type": "quantitative"},
                            "y": {"field": "Experimental HV", "type": "quantitative"},
                            "text": {"field": "Candidate", "type": "nominal"},
                        },
                    },
                    {"mark": {"type": "rule", "strokeDash": [6, 4]}, "encoding": {"x": {"datum": HARDNESS_REQUIREMENT}}},
                    {"mark": {"type": "rule", "strokeDash": [6, 4]}, "encoding": {"y": {"datum": HARDNESS_REQUIREMENT}}},
                ],
            },
            use_container_width=True,
        )
        st.caption(
            "The dashed lines mark 450 HV. The lower-right quadrant contains false passes: "
            "the model predicts PASS, but experiment says Fail."
        )

        false_pass = st.multiselect(
            "Which candidates are FALSE PASSES — ML says PASS, experiment says Fail?",
            pred.loc[pred["ML HV"] >= HARDNESS_REQUIREMENT, "Candidate"].tolist(),
            max_selections=2,
            key="s4_false",
        )

        st.subheader("2 · Error detective — four students, four different cases")
        st.markdown(
            '<div class="role-box"><strong>Split the cases:</strong> Student A investigates <strong>H</strong> · '
            'Student B investigates <strong>I</strong> · Student C investigates <strong>O</strong> · '
            'Student D investigates <strong>F</strong>. Spend 2–3 minutes on your case, then report back to the group. '
            'Do not assume that every large error has the same cause.</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "For each case, the app shows (i) prediction error, (ii) whether its composition is familiar to the model, "
            "and (iii) the five nearest training alloys in the full six-element composition space. "
            "**Nearest** here means smallest Euclidean distance in Al–Co–Cr–Cu–Fe–Ni at.% space."
        )

        case_descriptions = {
            "H": {
                "title": "H — large overprediction despite familiar chemistry",
                "prompt": (
                    "H is useful because a simple 'it was OOD' explanation does not work well. "
                    "Ask: if nominal composition is reasonably familiar, why can nearby compositions still have very different hardness?"
                ),
            },
            "I": {
                "title": "I — a moderate numerical error that changes the engineering decision",
                "prompt": (
                    "I demonstrates that error size and engineering consequence are not the same thing. "
                    "Ask: how can an error of only a few tens of HV turn a predicted PASS into an experimental FAIL?"
                ),
            },
            "O": {
                "title": "O — well-covered chemistry, but the experiment reaches a harder extreme",
                "prompt": (
                    "O asks you to think about prediction at the high end of the property range. "
                    "Ask: can an in-distribution composition still have a response beyond what its nearest training examples suggest?"
                ),
            },
            "F": {
                "title": "F — unusual chemistry, yet a small prediction error",
                "prompt": (
                    "F prevents a different oversimplification. "
                    "Ask: does being outside or near the edge of the training domain guarantee a bad prediction?"
                ),
            },
        }

        tabs = st.tabs([case_descriptions[c]["title"] for c in ["H", "I", "O", "F"]])

        for tab, cand_name in zip(tabs, ["H", "I", "O", "F"]):
            with tab:
                row = pred.loc[pred["Candidate"] == cand_name].iloc[0]
                diag = domain_diagnostics(row)
                neigh = nearest_training_examples(row, k=5)
                signed_error = float(row["ML HV"] - row["HV"])
                abs_error = abs(signed_error)
                local_min = float(neigh["Experimental HV"].min())
                local_max = float(neigh["Experimental HV"].max())

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ML prediction", f"{float(row['ML HV']):.0f} HV")
                m2.metric("Experiment", f"{float(row['HV']):.0f} HV")
                m3.metric("Prediction error", f"{signed_error:+.0f} HV")
                m4.metric("Coverage", diag["label"])

                st.info(case_descriptions[cand_name]["prompt"])

                st.markdown("**Five nearest training alloys by nominal composition**")
                st.dataframe(
                    neigh.style.format(
                        {
                            "Al": "{:.1f}",
                            "Co": "{:.1f}",
                            "Cr": "{:.1f}",
                            "Cu": "{:.1f}",
                            "Fe": "{:.1f}",
                            "Ni": "{:.1f}",
                            "Experimental HV": "{:.0f}",
                            "Distance": "{:.2f}",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    f"The five neighbours span {local_min:.0f}–{local_max:.0f} HV. "
                    "Compositionally nearby does not necessarily mean identical microstructure or identical hardness."
                )

                if cand_name == "H":
                    st.markdown(
                        '<div class="takeaway"><strong>Evidence to discuss:</strong> H is classified as reasonably well covered, '
                        'yet its error is large and its nearby training alloys span a wide hardness range. '
                        'So OOD alone cannot explain the miss. A plausible interpretation is that nominal composition does not uniquely encode '
                        'all hardness-controlling information—phase constitution/fractions, ordering, segregation, grain structure, defects or processing-linked microstructure may differ. '
                        '<strong>This is a hypothesis from the evidence, not proof of which hidden variable caused H.</strong></div>',
                        unsafe_allow_html=True,
                    )
                elif cand_name == "I":
                    st.markdown(
                        '<div class="takeaway"><strong>Evidence to discuss:</strong> I is only moderately mispredicted numerically, '
                        'but the error crosses the 450-HV specification. This is a decision-boundary problem: '
                        '<strong>a modest regression error can become a serious engineering classification error when the prediction is close to the requirement.</strong></div>',
                        unsafe_allow_html=True,
                    )
                elif cand_name == "O":
                    st.markdown(
                        '<div class="takeaway"><strong>Evidence to discuss:</strong> O is not obviously OOD, yet its experimental hardness lies above its nearest training examples. '
                        'Extra Trees predictions are built from averages of observed training responses in tree leaves, so the prediction remains anchored to hardness values represented in the training data rather than discovering a new physical extrapolation. '
                        '<strong>In-distribution does not mean exact, and a model can underpredict an unusually strong response.</strong></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="takeaway"><strong>Evidence to discuss:</strong> F is compositionally unusual by the coverage diagnostic, '
                        'yet its numerical prediction error is small. Therefore <strong>OOD is a risk flag, not a guarantee of failure</strong>. '
                        'Coverage tells us when extra caution is warranted; it does not deterministically predict the error of every individual sample.</div>',
                        unsafe_allow_html=True,
                    )

        st.subheader("3 · Group synthesis — match the evidence to the lesson")

        h_lesson = st.radio(
            "Case H: what is the strongest conclusion supported by the evidence?",
            [
                "Choose one…",
                "H proves that OOD chemistry is the cause of every large model error.",
                "A large error can occur even for familiar nominal chemistry; nearby alloys can have widely different hardness, suggesting that composition alone may omit important hardness-controlling information.",
                "Because H is in-domain, its experimental measurement must be wrong.",
            ],
            key="s4_h_lesson",
        )

        i_lesson = st.radio(
            "Case I: what is the main engineering lesson?",
            [
                "Choose one…",
                "Only very large numerical errors matter.",
                "A moderate numerical error can be decision-critical when the prediction lies near the 450-HV requirement.",
                "Any prediction above 450 HV should automatically be accepted.",
            ],
            key="s4_i_lesson",
        )

        o_lesson = st.radio(
            "Case O: what does this case show?",
            [
                "Choose one…",
                "Being well represented by the training compositions guarantees a small property error.",
                "An in-domain composition can still be underpredicted, particularly when the real response is more extreme than nearby observed training responses.",
                "PCA distance is a direct hardness equation.",
            ],
            key="s4_o_lesson",
        )

        f_lesson = st.radio(
            "Case F: what does this case show about OOD diagnostics?",
            [
                "Choose one…",
                "OOD or sparse coverage is a warning about reliability, not a guarantee that the prediction will be wrong.",
                "Every OOD prediction must have a large error.",
                "Training-domain coverage is irrelevant and should never be checked.",
            ],
            key="s4_f_lesson",
        )

        missing_info = st.multiselect(
            "Which types of information could plausibly affect hardness but are NOT explicitly supplied to this composition-only model?",
            [
                "Phase constitution and phase fractions",
                "Chemical ordering / precipitates",
                "Grain size and morphology",
                "Segregation and local chemical heterogeneity",
                "Defects and processing-linked microstructure",
                "Experimental/material variability",
                "The candidate letter (A, B, C, …)",
            ],
            key="s4_missing_info",
        )

        reflection = st.text_area(
            "Group conclusion (3–5 sentences): Why can no single explanation account for every prediction error? Use at least TWO of H, I, O or F as evidence.",
            key="s4_reflection",
            height=125,
            placeholder=(
                "For example: H shows that ... whereas F shows that ... . "
                "Therefore training-domain distance is useful because ..., but it cannot by itself ..."
            ),
        )

        if st.button("Complete Stage 4"):
            desired_missing = {
                "Phase constitution and phase fractions",
                "Chemical ordering / precipitates",
                "Grain size and morphology",
                "Segregation and local chemical heterogeneity",
                "Defects and processing-linked microstructure",
                "Experimental/material variability",
            }
            if set(false_pass) != {"H", "I"}:
                st.error("Identify the two predicted passes whose experimental hardness is below 450 HV.")
            elif not h_lesson.startswith("A large error can occur"):
                st.error("For H, use the coverage and neighbour evidence: a large miss occurs even though the nominal chemistry is not obviously OOD.")
            elif not i_lesson.startswith("A moderate numerical error"):
                st.error("For I, focus on the consequence of crossing the 450-HV decision boundary.")
            elif not o_lesson.startswith("An in-domain composition"):
                st.error("For O, distinguish being in-domain from being guaranteed accurate.")
            elif not f_lesson.startswith("OOD or sparse coverage"):
                st.error("For F, remember that OOD is a risk warning, not a deterministic error detector.")
            elif set(missing_info) != desired_missing:
                st.error("Select the physically plausible missing information, but not the candidate letter.")
            elif len(reflection.strip()) < 100:
                st.error("Write a short group synthesis using at least two of the four cases as evidence.")
            else:
                st.session_state.s4_complete = True
                st.success("Stage 4 complete.")
                v1, v2, v3 = st.columns(3)
                v1.metric("15-alloy MAE", "29.8 HV")
                v2.metric("15-alloy RMSE", "37.7 HV")
                v3.metric("15-alloy R²", "0.962")
                st.markdown(
                    '<div class="takeaway"><strong>Core intuition:</strong> prediction error is not one phenomenon with one cause. '
                    'Coverage, decision margin, local ambiguity, missing physical variables, model structure and experimental variability can matter in different ways. '
                    'A strong average R² therefore does not guarantee a safe individual engineering decision.</div>',
                    unsafe_allow_html=True,
                )

    stage_nav(4)


# ============================================================
# STAGE 5 — RICH MATERIALS-SELECTION CHALLENGE
# ============================================================

elif st.session_state.stage == 5:
    st.markdown('<div class="stage-title">Stage 5 — Materials-selection challenge</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aim-box"><strong>Aim:</strong> move beyond “pick the hardest alloy.” Eliminate infeasible options, identify dominated choices, debate the genuine trade-off, and show how the best material changes when the design brief changes.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Design brief")
    st.write(
        "Your team must recommend an alloy for a wear-resistant component. It must satisfy **experimental hardness ≥ 450 HV**. "
        "Low density is desirable and raw-material cost matters. There is no single universal scoring formula."
    )

    st.markdown(
        '<div class="role-box"><strong>Switch roles now:</strong> Student A = performance advocate · Student B = cost advocate · Student C = lightweight-design advocate · Student D = decision chair. Each advocate must make a quantitative case before the chair records the group decision.</div>',
        unsafe_allow_html=True,
    )

    if "HV" not in candidates.columns:
        st.error("The instructor candidate CSV must include experimental `HV` for the final selection stage.")
        st.stop()

    feasible = candidates.loc[candidates["HV"] >= HARDNESS_REQUIREMENT].copy()
    feasible = add_selection_metrics(feasible, candidates)

    st.subheader("A. Feasibility and selection metrics")
    st.markdown(
        r"""
**Experimental hardness** is the gate: a material below 450 HV is infeasible.

**Hardness margin** measures robustness above the requirement:

\[
\text{margin}=HV_{exp}-450
\]

**Density** matters when component mass matters. We estimate ideal mixture density using:

\[
\rho_{alloy}=\frac{\sum_i x_iM_i}{\sum_i x_iM_i/\rho_i}
\]

**Specific hardness** is a simple classroom comparison of hardness relative to density:

\[
\frac{HV}{\rho}
\]

**Relative cost** is a transparent teaching index based on elemental mass fractions and fixed teaching weights. It is **not a live market-price estimate**.
"""
    )

    const = pd.DataFrame(
        {
            "Element": ELEMENTS,
            "Atomic mass (g/mol)": [ATOMIC_MASS[e] for e in ELEMENTS],
            "Element density (g/cm³)": [ELEMENT_DENSITY[e] for e in ELEMENTS],
            "Teaching cost weight": [COST_WEIGHT[e] for e in ELEMENTS],
        }
    )
    with st.expander("Show constants used for density and teaching-cost calculations"):
        st.dataframe(const, hide_index=True, use_container_width=True)
        st.caption("Cost weights: Fe 1, Al 2, Cr 2.5, Cu 8, Ni 10, Co 25. These are pedagogical weights only.")

    table = feasible[["Candidate", "HV", "Hardness margin", "Density", "Specific hardness", "Relative cost"]].copy()
    table = table.sort_values("Candidate")
    st.dataframe(
        table.style.format(
            {
                "HV": "{:.0f}",
                "Hardness margin": "{:+.0f}",
                "Density": "{:.2f}",
                "Specific hardness": "{:.1f}",
                "Relative cost": "{:.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    trade1, trade2 = st.tabs(["Cost ↔ hardness", "Density ↔ hardness"])
    selection_values = table.to_dict(orient="records")
    with trade1:
        st.vega_lite_chart(
            {
                "height": 340,
                "data": {"values": selection_values},
                "layer": [
                    {
                        "mark": {"type": "point", "filled": True, "size": 150},
                        "encoding": {
                            "x": {"field": "Relative cost", "type": "quantitative", "title": "Relative teaching cost (lower is better)"},
                            "y": {"field": "HV", "type": "quantitative", "title": "Experimental hardness (higher is better)"},
                            "tooltip": [
                                {"field": "Candidate", "type": "nominal"},
                                {"field": "Relative cost", "type": "quantitative", "format": ".2f"},
                                {"field": "HV", "type": "quantitative", "format": ".0f"},
                            ],
                        },
                    },
                    {
                        "mark": {"type": "text", "dy": -12, "fontSize": 13},
                        "encoding": {
                            "x": {"field": "Relative cost", "type": "quantitative"},
                            "y": {"field": "HV", "type": "quantitative"},
                            "text": {"field": "Candidate", "type": "nominal"},
                        },
                    },
                ],
            },
            use_container_width=True,
        )
        st.caption("The desirable direction is toward the upper-left: higher hardness at lower cost.")
    with trade2:
        st.vega_lite_chart(
            {
                "height": 340,
                "data": {"values": selection_values},
                "layer": [
                    {
                        "mark": {"type": "point", "filled": True, "size": 150},
                        "encoding": {
                            "x": {"field": "Density", "type": "quantitative", "title": "Density (g/cm³; lower is better)"},
                            "y": {"field": "HV", "type": "quantitative", "title": "Experimental hardness (higher is better)"},
                            "tooltip": [
                                {"field": "Candidate", "type": "nominal"},
                                {"field": "Density", "type": "quantitative", "format": ".2f"},
                                {"field": "HV", "type": "quantitative", "format": ".0f"},
                                {"field": "Specific hardness", "type": "quantitative", "format": ".1f"},
                            ],
                        },
                    },
                    {
                        "mark": {"type": "text", "dy": -12, "fontSize": 13},
                        "encoding": {
                            "x": {"field": "Density", "type": "quantitative"},
                            "y": {"field": "HV", "type": "quantitative"},
                            "text": {"field": "Candidate", "type": "nominal"},
                        },
                    },
                ],
            },
            use_container_width=True,
        )
        st.caption("Again the desirable direction is upper-left: harder while lighter. Notice how this changes the apparent attractiveness of the candidates.")

    st.subheader("B. Eliminate dominated candidates")
    st.write(
        "A candidate is **dominated** if another feasible alloy is at least as good in all three of these directions — higher hardness, lower density, and lower cost — and strictly better in at least one. "
        "A dominated material has no reason to remain on the shortlist under this design brief."
    )

    dominated_truth = {}
    for cand in feasible["Candidate"]:
        by = dominated_by(feasible, cand)
        if by is not None:
            dominated_truth[str(cand)] = by

    dominated_choice = st.multiselect(
        "Which candidates are dominated and can be removed before the final debate?",
        feasible["Candidate"].tolist(),
        key="s5_dominated",
    )

    if st.button("Check dominance"):
        if set(dominated_choice) == set(dominated_truth.keys()):
            st.success("Correct. You have reduced the problem to the non-dominated frontier.")
            for cand, by in dominated_truth.items():
                st.write(f"- **{cand}** is dominated by **{by}**.")
            st.session_state.s5_dom_ok = True
        else:
            st.error("Re-check whether another candidate is simultaneously harder, lighter (lower density), and cheaper.")

    if st.session_state.get("s5_dom_ok", False):
        frontier = feasible.loc[~feasible["Candidate"].isin(dominated_truth.keys())].copy()
        st.markdown(
            '<div class="takeaway"><strong>Non-dominated shortlist:</strong> the meaningful trade-off collapses to <strong>L versus O</strong>. L is much cheaper while still comfortably feasible; O is substantially harder and lighter, but costs more.</div>',
            unsafe_allow_html=True,
        )

        st.subheader("C. Quantify L versus O")
        L = frontier.loc[frontier["Candidate"] == "L"].iloc[0]
        O = frontier.loc[frontier["Candidate"] == "O"].iloc[0]
        perf_gain = (O["HV"] / L["HV"] - 1) * 100
        cost_gain = (O["Relative cost"] / L["Relative cost"] - 1) * 100
        spec_gain = (O["Specific hardness"] / L["Specific hardness"] - 1) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("O hardness advantage vs L", f"+{O['HV']-L['HV']:.0f} HV", f"{perf_gain:.1f}%")
        c2.metric("O relative-cost increase vs L", f"+{O['Relative cost']-L['Relative cost']:.2f}", f"{cost_gain:.1f}%")
        c3.metric("O specific-hardness advantage", f"+{O['Specific hardness']-L['Specific hardness']:.1f}", f"{spec_gain:.1f}%")

        first_choice = st.radio(
            "After hearing the three advocates, which alloy does the group choose under the ORIGINAL brief?",
            ["Choose one…", "L", "O"],
            key="s5_first_choice",
        )
        first_reason = st.text_area(
            "Decision chair: defend the choice using at least three numerical comparisons, and explicitly say what you are giving up.",
            key="s5_first_reason",
            height=120,
        )

        st.subheader("D. Requirement change — does your answer survive?")
        scenario = st.selectbox(
            "Your supervisor changes the brief. Choose one scenario to test your recommendation:",
            [
                "Choose a scenario…",
                "Budget-first: material cost becomes the dominant consideration, but HV must remain ≥ 450.",
                "High-performance: minimum experimental hardness rises to 600 HV.",
                "Lightweight performance: maximise specific hardness among experimentally feasible candidates.",
                "Supply-risk: avoid Co-containing candidates if a suitable alternative exists.",
            ],
            key="s5_scenario",
        )

        new_choice = st.radio(
            "Under the changed brief, which finalist would you recommend?",
            ["Choose one…", "L", "O"],
            key="s5_new_choice",
        )
        new_reason = st.text_area(
            "Explain why the changed requirement does or does not change your recommendation.",
            key="s5_new_reason",
            height=100,
        )

        scenario_correct = False
        if scenario.startswith("Budget-first"):
            scenario_correct = new_choice == "L"
        elif scenario.startswith("High-performance"):
            scenario_correct = new_choice == "O"
        elif scenario.startswith("Lightweight performance"):
            scenario_correct = new_choice == "O"
        elif scenario.startswith("Supply-risk"):
            scenario_correct = new_choice == "L"

        st.subheader("E. Final engineering recommendation")
        rejected = "O" if first_choice == "L" else ("L" if first_choice == "O" else "")
        final_statement = st.text_area(
            "Write the final recommendation in 3–5 sentences: state your selected alloy, name the strongest alternative you rejected, and explain quantitatively why you rejected it.",
            key="s5_final_statement",
            height=130,
        )

        if st.button("Complete the practical", type="primary"):
            if first_choice == "Choose one…":
                st.error("Choose L or O under the original brief.")
            elif len(first_reason.strip()) < 100:
                st.error("Make the original decision justification more quantitative and complete.")
            elif scenario == "Choose a scenario…" or new_choice == "Choose one…":
                st.error("Complete the changed-brief decision.")
            elif not scenario_correct:
                st.error("Re-check the changed brief against the L/O data before finalising.")
            elif len(new_reason.strip()) < 50:
                st.error("Explain why the changed brief leads to that recommendation.")
            elif len(final_statement.strip()) < 100 or rejected not in final_statement:
                st.error("The final recommendation should explicitly mention the strongest rejected alternative and use quantitative evidence.")
            else:
                st.session_state.s5_complete = True
                st.success("Practical complete.")
                st.markdown(
                    r"""
<div class="takeaway"><strong>What you have done in one hour</strong><br><br>
You moved from <strong>composition → physical descriptors → researched hypotheses → ML prediction → uncertainty-aware screening → experimental validation → multi-objective engineering selection</strong>.<br><br>
The final lesson is not “ML replaces materials science.” It is:<br><br>
<strong>materials science helps you choose and interpret useful information;</strong><br>
<strong>ML can learn a more complex empirical mapping;</strong><br>
<strong>experiment tests whether the prediction is actually right;</strong><br>
<strong>engineering judgement decides what “best” means for a particular design brief.</strong>
</div>
""",
                    unsafe_allow_html=True,
                )

                summary = pd.DataFrame(
                    {
                        "Original recommendation": [first_choice],
                        "Changed brief": [scenario],
                        "Recommendation after change": [new_choice],
                        "Original justification": [first_reason.strip()],
                        "Changed-brief reasoning": [new_reason.strip()],
                        "Final statement": [final_statement.strip()],
                    }
                )
                st.download_button(
                    "Download group decision record (CSV)",
                    data=summary.to_csv(index=False).encode("utf-8"),
                    file_name="hea_group_decision_record.csv",
                    mime="text/csv",
                )

    stage_nav(5)


"""
app.py
======
Streamlit demo for Sleep Quality Score (SQS) prediction.

Run:
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
INPUT_DIR = Path("processed_4ch")

STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_COLORS = {
    "W":   "#E74C3C",
    "N1":  "#F39C12",
    "N2":  "#3498DB",
    "N3":  "#2C3E50",
    "REM": "#9B59B6",
}


# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sleep Quality Score",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    metadata = pd.read_csv(INPUT_DIR / "metadata.csv")
    quality = pd.read_csv(INPUT_DIR / "sleep_quality.csv")
    features = pd.read_csv(INPUT_DIR / "features.csv")
    preds = pd.read_csv(INPUT_DIR / "best_model_predictions.csv")
    imp = pd.read_csv(INPUT_DIR / "feature_importance.csv")
    return metadata, quality, features, preds, imp


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def sqs_color(score):
    if score >= 75:
        return "#27AE60"   # green
    elif score >= 55:
        return "#F39C12"   # orange
    return "#E74C3C"       # red


def sqs_label(score):
    if score >= 75:
        return "GOOD"
    elif score >= 55:
        return "FAIR"
    return "POOR"


def render_score_card(actual, predicted):
    color = sqs_color(predicted)
    label = sqs_label(predicted)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}44);
            border-left: 8px solid {color};
            border-radius: 12px;
            padding: 24px 32px;
            margin-bottom: 16px;
        ">
            <div style="font-size: 16px; color: #555; margin-bottom: 4px;">
                Predicted Sleep Quality Score
            </div>
            <div style="font-size: 64px; font-weight: bold; color: {color}; line-height: 1;">
                {predicted:.1f}
            </div>
            <div style="font-size: 22px; font-weight: 600; color: {color}; margin-top: 8px;">
                {label}
            </div>
            <div style="font-size: 14px; color: #777; margin-top: 12px;">
                Actual SQS: <b>{actual:.1f}</b> &nbsp;|&nbsp;
                Absolute Error: <b>{abs(actual - predicted):.2f}</b> points
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_hypnogram(record_meta):
    """Plot sleep stage over time for one record."""
    # Map stage id to numeric for plotting (REM at top, W at bottom)
    y_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    y_vals = [y_map[s] for s in record_meta["stage_name"]]
    x_vals = record_meta["epoch_idx"].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="lines",
        line=dict(color="#2C3E50", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(52, 152, 219, 0.2)",
        hovertemplate="Epoch %{x}<br>Stage: %{customdata}<extra></extra>",
        customdata=record_meta["stage_name"].values,
    ))
    fig.update_layout(
        title="Hypnogram (sleep stages over time)",
        xaxis_title="Epoch (30 seconds)",
        yaxis=dict(
            title="Sleep Stage",
            tickmode="array",
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["W", "N1", "N2", "N3", "REM"],
        ),
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        plot_bgcolor="white",
    )
    return fig


def plot_stage_distribution(row):
    stages = ["N1", "N2", "N3", "REM"]
    values = [row[f"{s}_pct"] for s in stages]
    colors = [STAGE_COLORS[s] for s in stages]

    fig = go.Figure(data=[go.Pie(
        labels=stages,
        values=values,
        marker=dict(colors=colors),
        hole=0.55,
        textinfo="label+percent",
    )])
    fig.update_layout(
        title="Sleep stage distribution",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig


def plot_feature_importance(imp_df, top_n=12):
    top = imp_df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["gain_importance"],
        y=top["feature"],
        orientation="h",
        marker_color="#3498DB",
    ))
    fig.update_layout(
        title=f"Top {top_n} most important features",
        height=420,
        margin=dict(l=180, r=20, t=40, b=40),
        xaxis_title="Gain importance",
        plot_bgcolor="white",
    )
    return fig


def plot_metrics_bars(row):
    """Show key sleep metrics as horizontal bars vs ideal range."""
    metrics = [
        ("Sleep Efficiency", row["SE"] * 100, 85, "%", 100),
        ("N3 (%)", row["N3_pct"], 18, "%", 40),
        ("REM (%)", row["REM_pct"], 22, "%", 40),
        ("WASO (min)", row["WASO_min"], 20, "min", 120),
        ("SOL (min)", row["SOL_min"], 15, "min", 60),
    ]

    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    ideals = [m[2] for m in metrics]
    maxes = [m[4] for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Actual", x=labels, y=values,
        marker_color="#3498DB",
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        name="Ideal", x=labels, y=ideals,
        mode="markers",
        marker=dict(color="#27AE60", size=14, symbol="diamond"),
    ))
    fig.update_layout(
        title="Sleep metrics vs ideal values",
        height=340,
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis_title="Value",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    st.title("🧠 Sleep Quality Score Analyzer")
    st.markdown(
        "**Predicting sleep quality from a single EEG channel (Pz-Oz)**  \n"
        "Powered by LightGBM trained on the Sleep-EDF Expanded dataset"
    )
    st.divider()

    # ---- Load data ----
    try:
        metadata, quality, features, preds, imp = load_data()
    except FileNotFoundError as e:
        st.error(f"Missing data file: {e}")
        st.stop()

    # ---- Sidebar ----
    st.sidebar.header("Select a recording")

    session_filter = st.sidebar.radio(
        "Session type",
        options=["All", "cassette", "telemetry"],
        index=0,
    )

    if session_filter == "All":
        available = preds["record_id"].tolist()
    else:
        available = preds[preds["session"] == session_filter]["record_id"].tolist()

    if not available:
        st.sidebar.warning("No records for this filter.")
        st.stop()

    record_id = st.sidebar.selectbox("Recording ID", available)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Model info**\n\n"
        "- Model: LightGBM Regressor\n"
        "- Features: 21 (EEG Pz-Oz only)\n"
        "- Validation: GroupKFold on subject_id\n"
        "- Domain adaptation: telemetry weight = 3"
    )

    # ---- Get row data ----
    prow = preds[preds["record_id"] == record_id].iloc[0]
    qrow = quality[quality["record_id"] == record_id].iloc[0]
    mrow = metadata[metadata["record_id"] == record_id].sort_values("epoch_idx")

    # ---- Top summary ----
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.subheader(f"Recording: {record_id}")
        st.caption(
            f"Subject {prow['subject_id']}  |  "
            f"Night {int(qrow['night'])}  |  "
            f"Session: {prow['session']}"
        )

    with col2:
        st.metric("Sleep Efficiency", f"{qrow['SE']*100:.1f}%")
    with col3:
        st.metric("TST (min)", f"{qrow['TST_min']:.0f}")
    with col4:
        st.metric("WASO (min)", f"{qrow['WASO_min']:.0f}")

    # ---- Score card ----
    render_score_card(prow["sleep_quality_score"], prow["predicted_sqs"])

    # ---- Hypnogram + Stage distribution ----
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(plot_hypnogram(mrow), use_container_width=True)
    with c2:
        st.plotly_chart(plot_stage_distribution(qrow), use_container_width=True)

    # ---- Metrics bars ----
    st.plotly_chart(plot_metrics_bars(qrow), use_container_width=True)

    # ---- Feature importance ----
    st.subheader("What drove this prediction?")
    st.caption(
        "The model relies mainly on **delta power** (deep sleep) and "
        "**sigma power** (sleep spindles), both established markers of healthy sleep."
    )
    st.plotly_chart(plot_feature_importance(imp, top_n=12), use_container_width=True)

    # ---- Raw values table ----
    with st.expander("See all computed metrics for this recording"):
        table = pd.DataFrame({
            "Metric": [
                "TIB (min)", "TST (min)", "SE (%)", "SOL (min)", "WASO (min)",
                "N1 (%)", "N2 (%)", "N3 (%)", "REM (%)",
                "Arousals", "Transitions", "Fragmentation / h",
                "SQS (actual)", "SQS (predicted)",
            ],
            "Value": [
                f"{qrow['TIB_min']:.1f}",
                f"{qrow['TST_min']:.1f}",
                f"{qrow['SE']*100:.1f}",
                f"{qrow['SOL_min']:.1f}",
                f"{qrow['WASO_min']:.1f}",
                f"{qrow['N1_pct']:.1f}",
                f"{qrow['N2_pct']:.1f}",
                f"{qrow['N3_pct']:.1f}",
                f"{qrow['REM_pct']:.1f}",
                f"{int(qrow['arousal_count'])}",
                f"{int(qrow['transitions'])}",
                f"{qrow['fragmentation_per_h']:.2f}",
                f"{qrow['sleep_quality_score']:.2f}",
                f"{prow['predicted_sqs']:.2f}",
            ],
        })
        st.dataframe(table, hide_index=True, use_container_width=True)

    # ---- Footer ----
    st.divider()
    st.caption(
        "Built for the AI Factory Iran Hackathon  |  "
        "Dataset: Sleep-EDF Expanded (PhysioNet)"
    )


if __name__ == "__main__":
    main()
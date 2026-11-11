"""Local Streamlit dashboard for the German Credit fairness analysis."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "figures"
EXPECTED_FILES = [
    OUTPUT / "results.json",
    OUTPUT / "metrics.csv",
    OUTPUT / "group_metrics.csv",
    OUTPUT / "bootstrap_differences.csv",
]

st.set_page_config(
    page_title="German Credit Fairness Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("German Credit Fairness Analysis")
st.caption(
    "Local dashboard for the UCI German Credit experiment · "
    "Women are the unprivileged group; good credit is the favorable outcome."
)


def german_fairness() -> None:
    with st.spinner("Running the analysis. This may take a minute..."):
        result = subprocess.run(
            [sys.executable, str(ROOT / "german_fairness.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        st.error("Analysis did not complete successfully.")
        if result.stdout:
            st.code(result.stdout)
        if result.stderr:
            st.code(result.stderr)
        st.stop()
    st.success("Analysis completed. The dashboard has refreshed.")
    if result.stdout:
        with st.expander("Analysis run details"):
            st.code(result.stdout)
    st.rerun()


missing_files = [path for path in EXPECTED_FILES if not path.is_file()]
if missing_files:
    st.info("Run the analysis to create the results shown in this dashboard.")
    if st.button("Run analysis now", type="primary"):
        german_fairness()
    st.stop()

if st.sidebar.button("Re-run analysis", type="primary"):
    german_fairness()

metrics = pd.read_csv(OUTPUT / "metrics.csv").set_index("model")
groups = pd.read_csv(OUTPUT / "group_metrics.csv")
bootstrap = pd.read_csv(OUTPUT / "bootstrap_differences.csv")

baseline = metrics.loc["Logistic Regression (baseline)"]
mitigated = metrics.loc["Logistic Regression (Reweighing)"]
overview, groups_tab, eda_tab, report_tab = st.tabs(
    ["Overview", "Group comparisons", "Explore the data", "Full report"]
)

with overview:
    st.subheader("Held-out test performance")
    performance_columns = st.columns(5)
    for column, metric, label in zip(
        performance_columns,
        ["accuracy", "precision", "recall", "f1", "roc_auc"],
        ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"],
    ):
        column.metric(
            label,
            f"{mitigated[metric]:.3f}",
            delta=f"{mitigated[metric] - baseline[metric]:+.3f} vs baseline",
        )

    st.subheader("Fairness metrics")
    st.caption(
        "SPD, EOD, and AOD are women minus men (parity = 0). "
        "DI is the women-to-men favorable prediction rate ratio (parity = 1)."
    )
    fairness_columns = st.columns(4)
    for column, metric, label in zip(
        fairness_columns,
        ["spd", "di", "eod", "aod"],
        ["SPD", "DI", "EOD", "AOD"],
    ):
        column.metric(
            label,
            f"{mitigated[metric]:.3f}",
            delta=f"{mitigated[metric] - baseline[metric]:+.3f} vs baseline",
        )

    st.image(
        str(FIGURES / "fairness_comparison.png"),
        caption="Logistic regression baseline compared with AIF360 Reweighing. The chart also shows random forest and gradient boosting comparisons.",
        use_container_width=True,
    )
    st.warning(
        "These results describe one split of a small historical benchmark. "
        "They are not evidence that the model is suitable for real credit decisions."
    )

    st.subheader("Paired bootstrap intervals")
    st.caption(
        "Descriptive 95% intervals for mitigated-minus-baseline test-set changes; "
        "they are not adjusted for multiple comparisons."
    )
    st.dataframe(bootstrap, hide_index=True, use_container_width=True)

with groups_tab:
    st.subheader("Performance by protected group")
    selected_model = st.selectbox("Model", groups["model"].drop_duplicates().tolist())
    group_data = groups.loc[groups["model"] == selected_model].set_index("group")
    st.dataframe(group_data, use_container_width=True)
    chart_data = group_data[["accuracy", "precision", "recall", "f1", "roc_auc"]].T
    st.bar_chart(chart_data, y_label="Score", x_label="Metric")
    st.caption(
        "The test split contains 140 men and 60 women. Small subgroup counts make "
        "group-level estimates uncertain."
    )

with eda_tab:
    st.subheader("Exploratory data analysis")
    eda_figures = [
        ("eda_outcome_by_sex.png", "Observed favorable-label rate by sex"),
        ("eda_age_by_outcome.png", "Age distribution by credit class"),
        ("eda_credit_amount.png", "Credit amount distribution"),
        ("eda_numeric_correlations.png", "Correlation matrix of numeric features"),
    ]
    for filename, caption in eda_figures:
        image_path = FIGURES / filename
        if image_path.is_file():
            st.image(str(image_path), caption=caption, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download descriptive statistics",
            data=(OUTPUT / "descriptive_statistics.csv").read_bytes(),
            file_name="descriptive_statistics.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "Download missing-value summary",
            data=(OUTPUT / "missing_values.csv").read_bytes(),
            file_name="missing_values.csv",
            mime="text/csv",
        )

with report_tab:
    report_path = OUTPUT / "REPORT.md"
    if report_path.is_file():
        report_text = report_path.read_text(encoding="utf-8")
        report_lines = [
            line for line in report_text.splitlines()
            if not line.startswith("![") and line.strip() != r"\newpage"
        ]
        st.markdown("\n".join(report_lines))
        st.download_button(
            "Download report (Markdown)",
            data=report_text,
            file_name="REPORT.md",
            mime="text/markdown",
        )
    else:
        st.info("The full report is not available yet. Re-run the analysis.")

presentation_path = OUTPUT / "presentation.pptx"
if presentation_path.is_file():
    st.sidebar.download_button(
        "Download presentation",
        data=presentation_path.read_bytes(),
        file_name="presentation.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
st.sidebar.caption("Runs locally in your browser; no dashboard data is uploaded.")

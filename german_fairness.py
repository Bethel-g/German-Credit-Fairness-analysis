"""Reproducible German Credit fairness analysis using IBM AIF360."""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "raw" / "german.data"
OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "figures"
DATA_URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
RANDOM_STATE = 42
TEST_SIZE = 0.20
BOOTSTRAP_REPLICATES = 1000
FAVORABLE_LABEL = 1
UNFAVORABLE_LABEL = 0
UNPRIVILEGED = [{"sex": 1.0}]
PRIVILEGED = [{"sex": 0.0}]

COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_rate",
    "personal_status_sex",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "credit_class",
]

SEX_FROM_STATUS = {
    "A91": 0,
    "A92": 1,
    "A93": 0,
    "A94": 0,
    "A95": 1,
}

NAVY = RGBColor(20, 42, 72)
BLUE = RGBColor(40, 105, 165)
TEAL = RGBColor(31, 145, 130)
PALE = RGBColor(239, 244, 249)
DARK = RGBColor(42, 52, 63)
MUTED = RGBColor(100, 112, 124)


def load_data() -> pd.DataFrame:
    """Load and validate the original whitespace-delimited UCI data."""
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_DATA.exists():
        request = urllib.request.Request(
            DATA_URL, headers={"User-Agent": "German-Credit-Fairness-Analysis/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = next(
                (name for name in archive.namelist() if name.endswith("german.data")),
                None,
            )
            if member is None:
                raise ValueError("The UCI archive did not contain german.data.")
            RAW_DATA.write_bytes(archive.read(member))

    data = pd.read_csv(RAW_DATA, sep=r"\s+", header=None, names=COLUMNS)
    if len(data) != 1000 or not set(data["credit_class"].unique()).issubset({1, 2}):
        raise ValueError("Unexpected German Credit data shape or target labels.")
    if not set(data["personal_status_sex"].unique()).issubset(SEX_FROM_STATUS):
        raise ValueError("Unexpected personal_status_sex code in source data.")

    data["sex"] = data["personal_status_sex"].map(SEX_FROM_STATUS).astype(int)
    data["credit_good"] = (data["credit_class"] == 1).astype(int)
    return data


def save_eda(data: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data.describe(include="all").transpose().to_csv(OUTPUT / "descriptive_statistics.csv")
    data.isna().sum().rename("missing_count").to_csv(OUTPUT / "missing_values.csv")
    data.groupby("sex", observed=True)["credit_good"].agg(
        ["count", "mean", "sum"]
    ).rename(index={0: "Men (privileged)", 1: "Women (unprivileged)"}).to_csv(
        OUTPUT / "outcome_by_sex.csv"
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rates = data.groupby("sex", observed=True)["credit_good"].mean().reindex([0, 1])
    ax.bar(["Men (privileged)", "Women (unprivileged)"], rates, color=["#2869a5", "#e08b48"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share with good-credit label")
    ax.set_title("Observed favorable-label rate by sex")
    for index, value in enumerate(rates):
        ax.text(index, value + 0.025, f"{value:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_outcome_by_sex.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color, name in [(1, "#2869a5", "Good credit"), (2, "#e08b48", "Bad credit")]:
        ax.hist(
            data.loc[data["credit_class"] == label, "age"],
            bins=18,
            alpha=0.58,
            color=color,
            label=name,
        )
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of applicants")
    ax.set_title("Age distribution by observed credit class")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_age_by_outcome.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(data["credit_amount"], bins=25, color="#318f82", edgecolor="white")
    ax.set_xlabel("Credit amount (DM)")
    ax.set_ylabel("Number of applicants")
    ax.set_title("Credit amount distribution")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_credit_amount.png", dpi=180)
    plt.close(fig)

    numeric = data.select_dtypes(include=np.number).drop(
        columns=["credit_class", "credit_good", "sex"], errors="ignore"
    )
    correlation = numeric.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(
        np.arange(len(correlation.columns)),
        correlation.columns,
        rotation=75,
        ha="right",
    )
    ax.set_yticks(np.arange(len(correlation.columns)), correlation.columns)
    ax.set_title("Correlation matrix of numeric features")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_numeric_correlations.png", dpi=180)
    plt.close(fig)


def make_dataset(features: pd.DataFrame, labels: pd.Series) -> BinaryLabelDataset:
    dataset_frame = features.copy()
    dataset_frame["credit_good"] = labels.to_numpy(dtype=float)
    return BinaryLabelDataset(
        df=dataset_frame,
        label_names=["credit_good"],
        protected_attribute_names=["sex"],
        favorable_label=float(FAVORABLE_LABEL),
        unfavorable_label=float(UNFAVORABLE_LABEL),
        privileged_protected_attributes=[[0.0]],
        unprivileged_protected_attributes=[[1.0]],
    )


def make_preprocessor(data: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    excluded = {"credit_class", "credit_good", "personal_status_sex"}
    feature_columns = [column for column in data.columns if column not in excluded]
    numeric = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(data[column]) and column != "sex"
    ]
    categorical = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(data[column]) and column != "sex"
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("protected", SimpleImputer(strategy="most_frequent"), ["sex"]),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=True,
    )
    return preprocessor, feature_columns


def transform_frame(
    preprocessor: ColumnTransformer, values: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    matrix = preprocessor.transform(values[columns])
    names = preprocessor.get_feature_names_out()
    frame = pd.DataFrame(matrix, columns=names, index=values.index)
    frame = frame.rename(columns={"protected__sex": "sex"})
    return frame.astype(float)


def group_performance(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, sex: np.ndarray
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for value, name in [(0, "Men (privileged)"), (1, "Women (unprivileged)")]:
        mask = sex == value
        y_group = labels[mask]
        pred_group = predictions[mask]
        prob_group = probabilities[mask]
        record: dict[str, Any] = {
            "group": name,
            "n": int(mask.sum()),
            "observed_good_rate": float(y_group.mean()),
            "predicted_good_rate": float(pred_group.mean()),
            "accuracy": float(accuracy_score(y_group, pred_group)),
            "precision": float(precision_score(y_group, pred_group, zero_division=0)),
            "recall": float(recall_score(y_group, pred_group, zero_division=0)),
            "f1": float(f1_score(y_group, pred_group, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_group, prob_group))
            if np.unique(y_group).size == 2
            else float("nan"),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def evaluate_model(
    model: LogisticRegression,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    sex_test: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    true_dataset = make_dataset(x_test, y_test)
    predicted_dataset = make_dataset(
        x_test, pd.Series(predictions, index=y_test.index, name="credit_good")
    )
    dataset_metric = BinaryLabelDatasetMetric(
        predicted_dataset,
        unprivileged_groups=UNPRIVILEGED,
        privileged_groups=PRIVILEGED,
    )
    classification_metric = ClassificationMetric(
        true_dataset,
        predicted_dataset,
        unprivileged_groups=UNPRIVILEGED,
        privileged_groups=PRIVILEGED,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "spd": float(dataset_metric.statistical_parity_difference()),
        "di": float(dataset_metric.disparate_impact()),
        "eod": float(classification_metric.equal_opportunity_difference()),
        "aod": float(classification_metric.average_odds_difference()),
    }
    group_table = group_performance(
        y_test.to_numpy(), predictions, probabilities, sex_test.to_numpy()
    )
    return metrics, group_table, predictions, probabilities


def bootstrap_differences(
    y: np.ndarray,
    sex: np.ndarray,
    baseline: np.ndarray,
    mitigated: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> pd.DataFrame:
    """Paired, sex-by-outcome stratified bootstrap for test-set metric changes."""
    rng = np.random.default_rng(RANDOM_STATE)
    metric_names = ["accuracy", "f1", "spd", "di", "eod", "aod"]
    draws: dict[str, list[float]] = {name: [] for name in metric_names}
    strata = [
        np.flatnonzero((sex == group) & (y == label))
        for group in (0, 1)
        for label in (0, 1)
    ]
    if any(len(indices) == 0 for indices in strata):
        raise ValueError("The test partition must contain all sex-by-outcome strata.")

    def calculate(labels: np.ndarray, protected: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        female = protected == 1
        male = protected == 0
        female_rate = float(pred[female].mean())
        male_rate = float(pred[male].mean())
        tpr_f = float(pred[female & (labels == 1)].mean())
        tpr_m = float(pred[male & (labels == 1)].mean())
        fpr_f = float(pred[female & (labels == 0)].mean())
        fpr_m = float(pred[male & (labels == 0)].mean())
        return {
            "accuracy": float(accuracy_score(labels, pred)),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "spd": female_rate - male_rate,
            "di": female_rate / male_rate if male_rate else float("nan"),
            "eod": tpr_f - tpr_m,
            "aod": 0.5 * ((tpr_f - tpr_m) + (fpr_f - fpr_m)),
        }

    for _ in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        )
        base_result = calculate(y[sampled], sex[sampled], baseline[sampled])
        mitigation_result = calculate(y[sampled], sex[sampled], mitigated[sampled])
        for name in metric_names:
            draws[name].append(mitigation_result[name] - base_result[name])

    rows = []
    for name, values in draws.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            low = median = high = float("nan")
        else:
            low, median, high = np.quantile(finite, [0.025, 0.5, 0.975])
        rows.append(
            {
                "metric": name,
                "delta_mitigated_minus_baseline": float(median),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        )
    return pd.DataFrame(rows)


def save_comparison_plot(
    baseline: dict[str, dict[str, float]], mitigated: dict[str, dict[str, float]]
) -> None:
    model_names = list(baseline.keys())
    colors = plt.get_cmap("tab10").colors
    width = 0.12

    fig, (ax, di_ax) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [3, 2]}
    )
    fairness_metrics = ["spd", "eod", "aod"]
    fairness_labels = ["SPD", "EOD", "AOD"]
    x = np.arange(len(fairness_metrics))
    for index, model_name in enumerate(model_names):
        color = colors[index]
        center = (index - (len(model_names) - 1) / 2) * 2 * width
        ax.bar(
            x + center - width / 2,
            [baseline[model_name][key] for key in fairness_metrics],
            width,
            color=color,
            alpha=0.4,
        )
        ax.bar(
            x + center + width / 2,
            [mitigated[model_name][key] for key in fairness_metrics],
            width,
            color=color,
            alpha=0.95,
        )
        di_ax.plot(
            [0, 1],
            [baseline[model_name]["di"], mitigated[model_name]["di"]],
            color=color,
            marker="o",
            label=model_name,
        )

    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xticks(x, fairness_labels)
    ax.set_ylabel("Difference (parity = 0)")
    ax.set_title("Group differences")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[index], label=name)
        for index, name in enumerate(model_names)
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8)
    ax.text(
        0.02,
        0.02,
        "Lighter bars: baseline; darker bars: Reweighing",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    di_ax.axhline(1, color="#666666", linewidth=0.8, linestyle="--")
    di_ax.set_xticks([0, 1], ["Baseline", "Reweighing"])
    di_ax.set_ylim(0.7, 1.1)
    di_ax.set_ylabel("Selection-rate ratio")
    di_ax.set_title("Disparate impact (zoomed scale)")
    di_ax.legend(frameon=False, fontsize=8)
    fig.suptitle("AIF360 fairness metrics on held-out test data")
    fig.tight_layout()
    fig.savefig(FIGURES / "fairness_comparison.png", dpi=180)
    plt.close(fig)

    perf_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    perf_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(perf_metrics))
    width = 0.1

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model_name in enumerate(model_names):
        center = (i - (len(model_names) - 1) / 2) * 2 * width
        color = colors[i]
        ax.bar(
            x + center - width / 2,
            [baseline[model_name][key] for key in perf_metrics],
            width,
            color=color,
            alpha=0.4,
        )
        ax.bar(
            x + center + width / 2,
            [mitigated[model_name][key] for key in perf_metrics],
            width,
            color=color,
            alpha=0.95,
        )

    ax.set_xticks(x, perf_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Predictive performance comparison across models")
    ax.legend(handles=handles, frameon=False, ncol=len(model_names), fontsize=8)
    ax.text(
        0.02,
        0.02,
        "Lighter bars: baseline; darker bars: Reweighing",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "performance_comparison.png", dpi=180)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return "not estimable" if not np.isfinite(value) else f"{value:.{digits}f}"


def create_report(
    data: pd.DataFrame,
    results: dict[str, Any],
    group_table: pd.DataFrame,
    bootstrap: pd.DataFrame,
    figure_prefix: str = "figures",
) -> str:
    """Render the project report from current metrics and the shared template."""
    template_path = ROOT / "documentation" / "REPORT_TEMPLATE.md"
    report = template_path.read_text(encoding="utf-8")
    model_names = list(results["baseline"])
    model_key = (
        "Logistic Regression"
        if "Logistic Regression" in model_names
        else model_names[0]
    )
    baseline = results["baseline"][model_key]
    mitigated = results["reweighing"][model_key]

    def group_row(model: str, group: str) -> pd.Series:
        match = group_table.loc[
            (group_table["model"] == model) & (group_table["group"] == group)
        ]
        if match.empty:
            raise ValueError(f"Missing group metrics for {model} / {group}.")
        return match.iloc[0]

    men_base = group_row(f"{model_key} (baseline)", "Men (privileged)")
    women_base = group_row(f"{model_key} (baseline)", "Women (unprivileged)")
    men_reweighed = group_row(f"{model_key} (Reweighing)", "Men (privileged)")
    women_reweighed = group_row(f"{model_key} (Reweighing)", "Women (unprivileged)")

    performance_lines = []
    for name in model_names:
        for method, metrics in (
            ("Baseline", results["baseline"][name]),
            ("Reweighing", results["reweighing"][name]),
        ):
            performance_lines.append(
                f"| {name} ({method}) | {metrics['accuracy']:.3f} | "
                f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                f"{metrics['f1']:.3f} | {metrics['roc_auc']:.3f} |"
            )

    group_lines = []
    for _, row in group_table.iterrows():
        group_lines.append(
            f"| {row['model']} / {row['group']} | {int(row['n'])} | "
            f"{row['observed_good_rate']:.3f} | {row['predicted_good_rate']:.3f} | "
            f"{row['accuracy']:.3f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} |"
        )

    metric_labels = [("SPD", "spd"), ("DI", "di"), ("EOD", "eod"), ("AOD", "aod")]
    primary_fairness = [
        f"| {label} | {baseline[key]:.3f} | {mitigated[key]:.3f} | "
        f"{mitigated[key] - baseline[key]:+.3f} |"
        for label, key in metric_labels
    ]
    all_fairness = []
    interpretations = []
    for name in model_names:
        before = results["baseline"][name]
        after = results["reweighing"][name]
        closer = []
        farther = []
        unchanged = []
        for label, key in metric_labels:
            parity = 1.0 if key == "di" else 0.0
            old_distance = abs(before[key] - parity)
            new_distance = abs(after[key] - parity)
            if new_distance < old_distance:
                closer.append(label)
            elif new_distance > old_distance:
                farther.append(label)
            else:
                unchanged.append(label)
            all_fairness.append(
                f"| {name} | {label} | {before[key]:.3f} | {after[key]:.3f} | "
                f"{after[key] - before[key]:+.3f} |"
            )
        changes = []
        if closer:
            changes.append(f"{', '.join(closer)} moved closer to parity")
        if farther:
            changes.append(f"{', '.join(farther)} moved farther from parity")
        if unchanged:
            changes.append(f"{', '.join(unchanged)} were unchanged")
        interpretations.append(f"- **{name}:** {'; '.join(changes)}.")

    bootstrap_lines = []
    for _, row in bootstrap.iterrows():
        low, high = row["ci95_low"], row["ci95_high"]
        interval = (
            f"[{low:+.3f}, {high:+.3f}]"
            if np.isfinite(low) and np.isfinite(high)
            else "not estimable"
        )
        bootstrap_lines.append(
            f"| {row.get('model', model_key)} | {row['metric'].upper()} | "
            f"{row['delta_mitigated_minus_baseline']:+.3f} | {interval} | "
            f"{'Yes' if row['ci_excludes_zero'] else 'No'} |"
        )

    primary_bootstrap = bootstrap.loc[bootstrap["model"] == model_key]
    bootstrap_by_metric = primary_bootstrap.set_index("metric")
    performance_interpretation = []
    for name in model_names:
        before = results["baseline"][name]
        after = results["reweighing"][name]
        model_bootstrap = bootstrap.loc[bootstrap["model"] == name].set_index(
            "metric"
        )
        accuracy_interval = model_bootstrap.loc["accuracy"]
        f1_interval = model_bootstrap.loc["f1"]
        performance_interpretation.append(
            f"- **{name}:** accuracy changed from {before['accuracy']:.3f} to "
            f"{after['accuracy']:.3f}; F1 from {before['f1']:.3f} to {after['f1']:.3f}; "
            f"ROC-AUC from {before['roc_auc']:.3f} to {after['roc_auc']:.3f}. "
            f"The accuracy interval "
            f"{'excludes' if accuracy_interval['ci_excludes_zero'] else 'includes'} zero; "
            f"the F1 interval "
            f"{'excludes' if f1_interval['ci_excludes_zero'] else 'includes'} zero."
        )

    primary_bootstrap_lines = []
    for metric in ("accuracy", "f1", "spd", "di", "eod", "aod"):
        if metric not in bootstrap_by_metric.index:
            continue
        row = bootstrap_by_metric.loc[metric]
        low, high = row["ci95_low"], row["ci95_high"]
        interval = (
            f"[{low:+.3f}, {high:+.3f}]"
            if np.isfinite(low) and np.isfinite(high)
            else "not estimable"
        )
        primary_bootstrap_lines.append(
            f"{metric.upper()} change was {row['delta_mitigated_minus_baseline']:+.3f} "
            f"(95% interval {interval}; "
            f"{'excludes' if row['ci_excludes_zero'] else 'includes'} zero)."
        )

    female_n = int((data["sex"] == 1).sum())
    male_n = int((data["sex"] == 0).sum())
    male_rate = float(data.loc[data["sex"] == 0, "credit_good"].mean())
    female_rate = float(data.loc[data["sex"] == 1, "credit_good"].mean())
    dataset_summary = (
        f"The recorded data contain {int(data['credit_good'].sum()):,} favorable labels "
        f"({data['credit_good'].mean():.1%}), {male_n} men and {female_n} women. "
        f"The observed favorable-label rates are {male_rate:.1%} for men and "
        f"{female_rate:.1%} for women. There are {int(data.isna().sum().sum())} missing "
        f"cells; median age is {data['age'].median():.1f} years and median credit amount "
        f"is {data['credit_amount'].median():,.0f} DM. These descriptive statistics "
        "characterize this dataset and are not estimates for a current lending population."
    )
    primary_results_summary = (
        f"The held-out test set contained {results['test_n']} applicants. For the "
        f"primary {model_key} model, accuracy changed from "
        f"{baseline['accuracy']:.3f} to {mitigated['accuracy']:.3f} and F1 from "
        f"{baseline['f1']:.3f} to {mitigated['f1']:.3f}. SPD changed from "
        f"{baseline['spd']:.3f} to {mitigated['spd']:.3f}, DI from "
        f"{baseline['di']:.3f} to {mitigated['di']:.3f}, EOD from "
        f"{baseline['eod']:.3f} to {mitigated['eod']:.3f}, and AOD from "
        f"{baseline['aod']:.3f} to {mitigated['aod']:.3f}. The interpretation is limited "
        "to this split and these particular metrics."
    )
    group_interpretation = (
        f"For {model_key}, women's test accuracy was {women_base['accuracy']:.3f} "
        f"versus men's {men_base['accuracy']:.3f} at baseline, and "
        f"{women_reweighed['accuracy']:.3f} versus {men_reweighed['accuracy']:.3f} "
        "after Reweighing. Women's predicted favorable rate changed from "
        f"{women_base['predicted_good_rate']:.3f} to "
        f"{women_reweighed['predicted_good_rate']:.3f}; men's changed from "
        f"{men_base['predicted_good_rate']:.3f} to "
        f"{men_reweighed['predicted_good_rate']:.3f}. The women-minus-men recall gap "
        f"changed from {women_base['recall'] - men_base['recall']:+.3f} to "
        f"{women_reweighed['recall'] - men_reweighed['recall']:+.3f}. "
        "These are descriptive estimates, with smaller subgroup counts for women."
    )
    test_group_counts = (
        f"The held-out test set contains {int(men_base['n'])} men and "
        f"{int(women_base['n'])} women."
    )
    primary_bootstrap_summary = " ".join(primary_bootstrap_lines)
    bootstrap_interpretation = (
        f"The primary-model intervals are summarized below: {primary_bootstrap_summary} "
        "For other models, the complete table shows the corresponding changes and intervals."
    )

    replacements = {
        "{{FIGURE_DIR}}": figure_prefix,
        "{{PRIMARY_RESULTS_SUMMARY}}": primary_results_summary,
        "{{DATASET_SUMMARY}}": dataset_summary,
        "{{TEST_GROUP_COUNTS}}": test_group_counts,
        "{{MODEL_PERFORMANCE}}": "\n".join(performance_lines),
        "{{PERFORMANCE_INTERPRETATION}}": "\n".join(performance_interpretation),
        "{{GROUP_INTERPRETATION}}": group_interpretation,
        "{{GROUP_TABLE}}": "\n".join(group_lines),
        "{{PRIMARY_FAIRNESS}}": "\n".join(primary_fairness),
        "{{ALL_MODEL_FAIRNESS}}": "\n".join(all_fairness),
        "{{MODEL_FAIRNESS_INTERPRETATION}}": "\n".join(interpretations),
        "{{BOOTSTRAP_TABLE}}": "\n".join(bootstrap_lines),
        "{{BOOTSTRAP_INTERPRETATION}}": bootstrap_interpretation,
    }
    for token, value in replacements.items():
        report = report.replace(token, value)
    if "{{" in report:
        raise ValueError("The report template contains an unresolved placeholder.")
    return report


def add_text(
    slide: Any,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    size: int = 18,
    color: RGBColor = DARK,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = align
    paragraph.font.name = "Aptos"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = color


def add_bullet_list(slide: Any, lines: list[str], x: float, y: float, w: float, h: float) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.level = 0
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(19)
        paragraph.font.color.rgb = DARK
        paragraph.space_after = Pt(12)


def add_slide_title(slide: Any, title: str, subtitle: str | None = None) -> None:
    add_text(slide, title, 0.65, 0.3, 12.0, 0.58, size=27, color=NAVY, bold=True)
    if subtitle:
        add_text(slide, subtitle, 0.68, 0.91, 12.0, 0.4, size=12, color=MUTED)
    line = slide.shapes.add_shape(1, Inches(0.67), Inches(1.38), Inches(12.0), Inches(0.025))
    line.fill.solid()
    line.fill.fore_color.rgb = TEAL
    line.line.fill.background()


def create_presentation(results: dict[str, Any], group_table: pd.DataFrame) -> None:
    """Create a 12-slide presentation from the recorded analysis outputs."""
    deck = Presentation()
    deck.slide_width = Inches(13.333)
    deck.slide_height = Inches(7.5)
    blank = deck.slide_layouts[6]
    white = RGBColor(255, 255, 255)
    pale = RGBColor(239, 244, 249)
    orange = RGBColor(224, 139, 72)

    def new_slide(title: str, subtitle: str | None = None) -> Any:
        slide = deck.slides.add_slide(blank)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = white
        add_slide_title(slide, title, subtitle)
        return slide

    def add_footer(slide: Any, number: int) -> None:
        add_text(
            slide,
            f"GERMAN CREDIT FAIRNESS ANALYSIS     •     {number:02d} / 12",
            8.15,
            7.08,
            4.5,
            0.2,
            size=9,
            color=MUTED,
            align=PP_ALIGN.RIGHT,
        )

    def add_card(
        slide: Any,
        x: float,
        y: float,
        w: float,
        h: float,
        heading: str,
        body: str,
        accent: RGBColor = TEAL,
    ) -> None:
        shape = slide.shapes.add_shape(
            1, Inches(x), Inches(y), Inches(w), Inches(h)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = pale
        shape.line.fill.background()
        bar = slide.shapes.add_shape(
            1, Inches(x), Inches(y), Inches(0.08), Inches(h)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent
        bar.line.fill.background()
        add_text(slide, heading, x + 0.22, y + 0.12, w - 0.4, 0.38, 17, NAVY, True)
        add_text(slide, body, x + 0.22, y + 0.57, w - 0.42, h - 0.68, 13, DARK)

    def add_metric_table(
        slide: Any,
        rows: list[list[str]],
        x: float,
        y: float,
        w: float,
        h: float,
        widths: list[float] | None = None,
        font_size: int = 12,
    ) -> None:
        table_shape = slide.shapes.add_table(
            len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)
        )
        table = table_shape.table
        if widths:
            for column, width in zip(table.columns, widths):
                column.width = Inches(width)
        for row_index, values in enumerate(rows):
            for column_index, value in enumerate(values):
                cell = table.cell(row_index, column_index)
                cell.text = value
                cell.margin_left = Inches(0.08)
                cell.margin_right = Inches(0.06)
                cell.margin_top = Inches(0.04)
                cell.margin_bottom = Inches(0.03)
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY if row_index == 0 else (
                    pale if row_index % 2 == 0 else white
                )
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.font.name = "Aptos"
                    paragraph.font.size = Pt(font_size)
                    paragraph.font.bold = row_index == 0
                    paragraph.font.color.rgb = white if row_index == 0 else DARK

    model_names = list(results["baseline"])
    model_key = "Logistic Regression" if "Logistic Regression" in model_names else model_names[0]
    baseline = results["baseline"][model_key]
    mitigated = results["reweighing"][model_key]

    # 1. Cover
    slide = deck.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    add_text(slide, "FAIRNESS IN CREDIT SCORING", 0.9, 0.95, 11.5, 0.45, 16, TEAL, True)
    add_text(
        slide,
        "Measuring and mitigating\ngroup disparities",
        0.9,
        1.8,
        11.5,
        1.6,
        36,
        white,
        True,
    )
    add_text(
        slide,
        "German Credit benchmark  |  IBM AI Fairness 360  |  Project report",
        0.94,
        3.72,
        11.3,
        0.48,
        17,
        RGBColor(218, 228, 238),
    )
    add_text(
        slide,
        "Betel Negusu  •  Esrom Adugna  •  Aklile Solomon",
        0.94,
        5.65,
        11.3,
        0.4,
        15,
        RGBColor(218, 228, 238),
    )
    add_text(
        slide,
        "Educational benchmark analysis — not a lending recommendation",
        0.94,
        6.25,
        11.3,
        0.38,
        12,
        TEAL,
    )

    # 2. Context and question
    slide = new_slide("Why audit credit classification?", "CONTEXT  /  QUESTION")
    add_card(
        slide,
        0.8,
        1.75,
        5.65,
        1.65,
        "Why it matters",
        "Credit access is consequential. Historical records and proxy features can carry patterns that affect groups differently.",
        TEAL,
    )
    add_card(
        slide,
        6.8,
        1.75,
        5.65,
        1.65,
        "Project question",
        "How do group selection rates and error rates compare, and how do they change after training-data reweighting?",
        orange,
    )
    add_text(slide, "WHAT THIS STUDY DOES", 0.9, 4.05, 3.0, 0.35, 13, TEAL, True)
    add_bullet_list(
        slide,
        [
            "Audits one historical benchmark with several performance and fairness measures.",
            "Compares baseline models with the same models trained using AIF360 Reweighing.",
            "Reports uncertainty and limits; observed associations are not causal findings.",
        ],
        0.9,
        4.42,
        11.6,
        1.9,
    )
    add_footer(slide, 2)

    # 3. Dataset and EDA
    slide = new_slide("Dataset snapshot", "DATA  /  EXPLORATORY ANALYSIS")
    slide.shapes.add_picture(
        str(FIGURES / "eda_outcome_by_sex.png"),
        Inches(0.75),
        Inches(1.7),
        width=Inches(5.1),
    )
    slide.shapes.add_picture(
        str(FIGURES / "eda_credit_amount.png"),
        Inches(6.6),
        Inches(1.7),
        width=Inches(5.8),
    )
    add_card(slide, 0.85, 5.15, 3.7, 1.2, "1,000 records", "20 model features • 700 good-credit labels")
    add_card(slide, 4.8, 5.15, 3.7, 1.2, "31% women", "310 women • 690 men in the full dataset", orange)
    add_card(slide, 8.75, 5.15, 3.7, 1.2, "No missing cells", "Median age 33 • median amount 2,320 DM")
    add_footer(slide, 3)

    # 4. Label and group definitions
    slide = new_slide("Define the outcome and audit groups", "OPERATIONAL DEFINITIONS")
    add_card(
        slide,
        0.85,
        1.8,
        5.55,
        2.0,
        "Favorable outcome",
        "Original class 1 (good credit) → credit_good = 1\nOriginal class 2 (bad credit) → credit_good = 0",
    )
    add_card(
        slide,
        6.85,
        1.8,
        5.55,
        2.0,
        "Protected attribute mapping",
        "Women: A92 / A95 (unprivileged)\nMen: A91 / A93 / A94 (privileged)",
        orange,
    )
    add_text(
        slide,
        "In the full dataset, observed good-credit labels were 64.8% for women and 72.3% for men.",
        1.05,
        4.35,
        11.1,
        0.55,
        20,
        NAVY,
        True,
        PP_ALIGN.CENTER,
    )
    add_text(
        slide,
        "These groups reflect the source field’s binary coding for this audit; they do not represent all sex or gender identities. Historical label differences are descriptive, not causal.",
        1.15,
        5.05,
        10.9,
        0.8,
        15,
        MUTED,
        align=PP_ALIGN.CENTER,
    )
    add_footer(slide, 4)

    # 5. Workflow
    slide = new_slide("Analysis workflow", "REPRODUCIBLE EXPERIMENT")
    workflow = [
        ("01", "Validate data", "1,000 rows, labels, group codes"),
        ("02", "Prepare features", "Impute, scale, one-hot encode"),
        ("03", "Split", "80/20 stratified • seed 42"),
        ("04", "Train", "3 model families • baseline + weighted"),
        ("05", "Evaluate", "Held-out performance + fairness"),
    ]
    for index, (number, heading, body) in enumerate(workflow):
        x = 0.72 + index * 2.52
        add_card(slide, x, 2.1, 2.2, 2.35, f"{number}  {heading}", body, TEAL if index != 3 else orange)
        if index < len(workflow) - 1:
            add_text(slide, "›", x + 2.23, 2.95, 0.25, 0.45, 26, MUTED, True, PP_ALIGN.CENTER)
    add_text(
        slide,
        "Preprocessing is fitted on training data only. The same held-out test set and 0.50 threshold are used for each paired comparison.",
        1.05,
        5.15,
        11.2,
        0.75,
        17,
        NAVY,
        align=PP_ALIGN.CENTER,
    )
    add_footer(slide, 5)

    # 6. Models and reweighing
    slide = new_slide("Models and mitigation", "WHAT WAS COMPARED")
    add_metric_table(
        slide,
        [
            ["Model", "Role in comparison"],
            ["Logistic regression", "Primary, interpretable reference model"],
            ["Random forest", "Ensemble comparison"],
            ["Gradient boosting", "Boosted-tree comparison"],
        ],
        0.85,
        1.85,
        5.45,
        2.75,
        [2.4, 3.05],
        15,
    )
    add_card(
        slide,
        6.8,
        1.85,
        5.55,
        2.75,
        "AIF360 Reweighing",
        "Fits weights on training examples to reduce dependence between group membership and observed outcome.\n\nEach classifier is retrained with those weights; test cases are not reweighted.",
        orange,
    )
    add_text(
        slide,
        "Six evaluated combinations: three model families × baseline and Reweighing.",
        0.95,
        5.2,
        11.3,
        0.5,
        18,
        NAVY,
        True,
        PP_ALIGN.CENTER,
    )
    add_text(
        slide,
        "Reweighing changes training influence; it does not guarantee parity or correct biased labels.",
        1.05,
        5.85,
        11.1,
        0.45,
        14,
        MUTED,
        align=PP_ALIGN.CENTER,
    )
    add_footer(slide, 6)

    # 7. Performance
    slide = new_slide("Held-out predictive performance", "N = 200  /  ACCURACY, F1 AND ROC-AUC")
    slide.shapes.add_picture(
        str(FIGURES / "performance_comparison.png"),
        Inches(0.65),
        Inches(1.65),
        width=Inches(7.0),
    )
    add_text(slide, "Primary model: logistic regression", 8.0, 1.75, 4.4, 0.4, 17, NAVY, True)
    add_metric_table(
        slide,
        [
            ["Metric", "Baseline", "Reweighing"],
            ["Accuracy", f"{baseline['accuracy']:.3f}", f"{mitigated['accuracy']:.3f}"],
            ["Precision", f"{baseline['precision']:.3f}", f"{mitigated['precision']:.3f}"],
            ["Recall", f"{baseline['recall']:.3f}", f"{mitigated['recall']:.3f}"],
            ["F1", f"{baseline['f1']:.3f}", f"{mitigated['f1']:.3f}"],
            ["ROC-AUC", f"{baseline['roc_auc']:.3f}", f"{mitigated['roc_auc']:.3f}"],
        ],
        7.95,
        2.3,
        4.55,
        3.35,
        [1.65, 1.4, 1.5],
        13,
    )
    add_text(
        slide,
        "Accuracy unchanged; F1 and ROC-AUC shifted slightly. One split is not a general performance guarantee.",
        0.9,
        6.3,
        11.7,
        0.42,
        14,
        MUTED,
        align=PP_ALIGN.CENTER,
    )
    add_footer(slide, 7)

    # 8. Fairness metrics
    slide = new_slide("Fairness results: logistic regression", "AIF360  /  WOMEN MINUS MEN")
    slide.shapes.add_picture(
        str(FIGURES / "fairness_comparison.png"),
        Inches(0.55),
        Inches(1.65),
        width=Inches(7.45),
    )
    fairness_table = [["Measure", "Baseline", "Reweighing", "Parity"]]
    for key, label, parity in (
        ("spd", "SPD", "0"),
        ("di", "DI", "1"),
        ("eod", "EOD", "0"),
        ("aod", "AOD", "0"),
    ):
        fairness_table.append(
            [label, f"{baseline[key]:.3f}", f"{mitigated[key]:.3f}", parity]
        )
    add_metric_table(
        slide,
        fairness_table,
        8.0,
        2.0,
        4.6,
        2.9,
        [1.15, 1.2, 1.35, 0.9],
        13,
    )
    add_text(
        slide,
        "SPD: selection-rate difference\nDI: selection-rate ratio\nEOD: true-positive-rate difference\nAOD: average true/false-positive-rate difference",
        8.05,
        5.15,
        4.5,
        1.15,
        13,
        DARK,
    )
    add_text(
        slide,
        "Parity references: 0 for differences; 1 for DI. Closer on this model does not mean all harms are resolved.",
        0.9,
        6.48,
        11.55,
        0.32,
        12,
        MUTED,
        align=PP_ALIGN.CENTER,
    )
    add_footer(slide, 8)

    # 9. Group-level performance
    slide = new_slide("Group-level performance", "PRIMARY MODEL  /  HELD-OUT TEST SET")
    group_lookup = {
        (row["model"], row["group"]): row for _, row in group_table.iterrows()
    }
    group_rows = [["Condition / group", "N", "Pred. good", "Accuracy", "Recall"]]
    for condition in ("baseline", "Reweighing"):
        for group in ("Men (privileged)", "Women (unprivileged)"):
            row = group_lookup[f"Logistic Regression ({condition})", group]
            group_rows.append(
                [
                    f"{condition.title()} / {group.split(' (')[0]}",
                    str(int(row["n"])),
                    f"{row['predicted_good_rate']:.3f}",
                    f"{row['accuracy']:.3f}",
                    f"{row['recall']:.3f}",
                ]
            )
    add_metric_table(
        slide,
        group_rows,
        0.9,
        1.85,
        7.35,
        2.75,
        [2.6, 0.8, 1.3, 1.3, 1.35],
        13,
    )
    add_card(
        slide,
        8.65,
        1.85,
        3.75,
        2.75,
        "What changed?",
        "Women’s predicted favorable rate: 0.667 → 0.700\n\nMen’s rate: 0.750 → 0.721\n\nWomen-minus-men recall gap: -0.115 → -0.035",
        orange,
    )
    add_text(
        slide,
        "The test split contains 140 men and 60 women. Small subgroup counts make group estimates uncertain.",
        1.0,
        5.25,
        11.3,
        0.55,
        16,
        NAVY,
        align=PP_ALIGN.CENTER,
    )
    add_text(
        slide,
        "Observed group differences are descriptive and may reflect both model behavior and label prevalence; they are not causal effects.",
        1.05,
        5.98,
        11.2,
        0.55,
        13,
        MUTED,
        align=PP_ALIGN.CENTER,
    )
    add_footer(slide, 9)

    # 10. Uncertainty and model variability
    slide = new_slide("Uncertainty and model dependence", "PAIRED, STRATIFIED BOOTSTRAP  /  1,000 RESAMPLES")
    add_text(slide, "Primary-model changes (Reweighing − baseline)", 0.9, 1.72, 8.0, 0.38, 17, NAVY, True)
    ci_table = [["Metric", "Change", "95% interval", "Excludes 0?"]]
    bootstrap_frame = pd.read_csv(OUTPUT / "bootstrap_differences.csv")
    primary_ci = bootstrap_frame.loc[bootstrap_frame["model"] == model_key]
    for _, row in primary_ci.iterrows():
        ci_table.append(
            [
                row["metric"].upper(),
                f"{row['delta_mitigated_minus_baseline']:+.3f}",
                f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]",
                "Yes" if row["ci_excludes_zero"] else "No",
            ]
        )
    add_metric_table(
        slide,
        ci_table,
        0.9,
        2.2,
        6.9,
        3.65,
        [1.3, 1.2, 2.5, 1.9],
        13,
    )
    add_card(
        slide,
        8.2,
        2.2,
        4.15,
        3.65,
        "Read cautiously",
        "Accuracy and F1 intervals include zero.\n\nThe four primary-model fairness-change intervals exclude zero in this resampling procedure.\n\nThese are descriptive intervals—not proof of fairness, practical importance, or a result that generalizes to other splits.",
        orange,
    )
    add_text(
        slide,
        "Gradient boosting moved farther from parity on all four fairness point estimates; effects differ by model family.",
        0.95,
        6.2,
        11.3,
        0.45,
        14,
        MUTED,
        align=PP_ALIGN.CENTER,
    )
    add_footer(slide, 10)

    # 11. Limitations and responsible use
    slide = new_slide("Limitations and responsible use", "WHAT THE EXPERIMENT CANNOT ESTABLISH")
    add_card(
        slide,
        0.85,
        1.75,
        5.6,
        1.6,
        "Historical, small benchmark",
        "1,000 records from a different era and context; labels are not complete ground truth for modern lending.",
        orange,
    )
    add_card(
        slide,
        6.8,
        1.75,
        5.6,
        1.6,
        "One split and fixed threshold",
        "A single 80/20 split cannot establish robust generalization; the 0.50 threshold was not optimized.",
    )
    add_card(
        slide,
        0.85,
        3.65,
        5.6,
        1.6,
        "Incomplete fairness picture",
        "No causal, calibration, individual, privacy, or full intersectional audit; parity metrics capture different goals.",
    )
    add_card(
        slide,
        6.8,
        3.65,
        5.6,
        1.6,
        "Governance before any use",
        "Require legal and stakeholder review, human oversight and appeal, independent validation, and monitored rollback criteria.",
        orange,
    )
    add_text(
        slide,
        "This analysis is an educational audit—not a recommendation to deploy a credit scoring model.",
        1.0,
        6.0,
        11.2,
        0.48,
        17,
        NAVY,
        True,
        PP_ALIGN.CENTER,
    )
    add_footer(slide, 11)

    # 12. Summary and reproducibility
    slide = new_slide("Takeaways and reproducibility", "CONCLUSION  /  PROJECT DELIVERABLES")
    add_bullet_list(
        slide,
        [
            f"Primary logistic regression: accuracy {baseline['accuracy']:.3f} → {mitigated['accuracy']:.3f}; SPD {baseline['spd']:.3f} → {mitigated['spd']:.3f}.",
            "Reweighing narrowed several primary-model gaps, but outcomes varied across model families.",
            "Fairness conclusions depend on metric, labels, group definition, threshold, and real-world context.",
        ],
        0.9,
        1.75,
        11.5,
        2.2,
    )
    add_card(
        slide,
        0.95,
        4.45,
        5.5,
        1.35,
        "Re-run analysis",
        "python german_fairness.py\nCreates metrics, figures, report, and presentation.",
    )
    add_card(
        slide,
        6.8,
        4.45,
        5.5,
        1.35,
        "Open dashboard",
        "python -m streamlit run app.py\nLocal dashboard; results are not uploaded.",
        orange,
    )
    add_text(
        slide,
        "Outputs include CSV/JSON results, EDA and comparison figures, a 15-page report, dashboard, and this presentation.",
        1.05,
        6.15,
        11.2,
        0.42,
        13,
        MUTED,
        align=PP_ALIGN.CENTER,
    )
    add_footer(slide, 12)

    deck.save(OUTPUT / "presentation.pptx")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_data()
    save_eda(data)

    y = data["credit_good"]
    sex = data["sex"]
    indices = data.index
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train = data.loc[train_idx]
    test = data.loc[test_idx]
    preprocessor, feature_columns = make_preprocessor(data)
    preprocessor.fit(train[feature_columns])
    x_train = transform_frame(preprocessor, train, feature_columns)
    x_test = transform_frame(preprocessor, test, feature_columns)
    y_train = train["credit_good"]
    y_test = test["credit_good"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE, subsample=0.8, max_depth=5
        ),
    }

    baseline_results: dict[str, dict[str, float]] = {}
    mitigated_results: dict[str, dict[str, float]] = {}
    all_group_tables = []
    model_predictions: dict[str, dict[str, np.ndarray]] = {}

    for model_name, model in models.items():
        # Train baseline model
        model.fit(x_train, y_train)
        baseline_metrics, baseline_groups, baseline_pred, _ = evaluate_model(
            model, x_test, y_test, test["sex"]
        )
        baseline_results[model_name] = baseline_metrics

        # Apply Reweighing mitigation
        train_dataset = make_dataset(x_train, y_train)
        reweigher = Reweighing(
            unprivileged_groups=UNPRIVILEGED,
            privileged_groups=PRIVILEGED,
        )
        reweighed_train = reweigher.fit_transform(train_dataset)
        mit_model = clone(model)
        mit_model.fit(
            x_train,
            y_train,
            sample_weight=reweighed_train.instance_weights,
        )
        mitigated_metrics, mitigated_groups, mitigated_pred, _ = evaluate_model(
            mit_model, x_test, y_test, test["sex"]
        )
        mitigated_results[model_name] = mitigated_metrics
        model_predictions[model_name] = {
            "baseline": baseline_pred,
            "reweighing": mitigated_pred,
        }

        baseline_label = f"{model_name} (baseline)"
        mitigated_label = f"{model_name} (Reweighing)"
        model_groups = pd.concat(
            [
                baseline_groups.assign(model=baseline_label),
                mitigated_groups.assign(model=mitigated_label),
            ],
            ignore_index=True,
        )[["model", "group", "n", "observed_good_rate", "predicted_good_rate",
           "accuracy", "precision", "recall", "f1", "roc_auc"]]
        all_group_tables.append(model_groups)

    # Aggregate group tables across models
    group_table = pd.concat(all_group_tables, ignore_index=True)
    group_table.to_csv(OUTPUT / "group_metrics.csv", index=False)

    results: dict[str, Any] = {
        "dataset": "UCI Statlog German Credit",
        "n_rows": int(len(data)),
        "n_features_before_encoding": int(len(feature_columns)),
        "test_n": int(len(test)),
        "split_seed": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "threshold": 0.5,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "protected_groups": {
            "unprivileged": "Women (sex=1)",
            "privileged": "Men (sex=0)",
        },
        "baseline": baseline_results,
        "reweighing": mitigated_results,
    }

    metrics_table = pd.DataFrame(
        [
            dict(model="Logistic Regression (baseline)", **baseline_results["Logistic Regression"]),
            dict(model="Logistic Regression (Reweighing)", **mitigated_results["Logistic Regression"]),
            dict(model="Random Forest (baseline)", **baseline_results["Random Forest"]),
            dict(model="Random Forest (Reweighing)", **mitigated_results["Random Forest"]),
            dict(model="Gradient Boosting (baseline)", **baseline_results["Gradient Boosting"]),
            dict(model="Gradient Boosting (Reweighing)", **mitigated_results["Gradient Boosting"]),
        ]
    )
    metrics_table.to_csv(OUTPUT / "metrics.csv", index=False)

    bootstrap_results = []
    for model_name in models:
        bootstrap = bootstrap_differences(
            y_test.to_numpy(dtype=int),
            test["sex"].to_numpy(dtype=int),
            model_predictions[model_name]["baseline"],
            model_predictions[model_name]["reweighing"],
        )
        bootstrap["model"] = model_name
        bootstrap_results.append(bootstrap)

    bootstrap = pd.concat(bootstrap_results, ignore_index=True)
    bootstrap.to_csv(OUTPUT / "bootstrap_differences.csv", index=False)

    save_comparison_plot(baseline_results, mitigated_results)
    (OUTPUT / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    report = create_report(data, results, group_table, bootstrap)
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")
    documentation_report = create_report(
        data, results, group_table, bootstrap, figure_prefix="../outputs/figures"
    )
    (ROOT / "documentation" / "REPORT.md").write_text(
        documentation_report, encoding="utf-8"
    )
    create_presentation(results, group_table)

    print(f"Analysis complete. Outputs: {OUTPUT}")
    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nAIF360 fairness metric differences use women minus men.")
    print(bootstrap.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()

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
    baseline: dict[str, float], mitigated: dict[str, float]
) -> None:
    names = ["SPD", "DI", "EOD", "AOD"]
    keys = ["spd", "di", "eod", "aod"]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, [baseline[key] for key in keys], width, label="Baseline", color="#2869a5")
    ax.bar(x + width / 2, [mitigated[key] for key in keys], width, label="Reweighing", color="#318f82")
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.axhline(1, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, names)
    ax.set_ylabel("Metric value")
    ax.set_title("AIF360 fairness metrics on held-out test data")
    ax.legend(frameon=False)
    ax.text(0.99, 0.02, "Parity references: 0 (SPD/EOD/AOD), 1 (DI)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(FIGURES / "fairness_comparison.png", dpi=180)
    plt.close(fig)

    names = ["Accuracy", "F1", "ROC-AUC"]
    keys = ["accuracy", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x[:3] - width / 2, [baseline[key] for key in keys], width, label="Baseline", color="#2869a5")
    ax.bar(x[:3] + width / 2, [mitigated[key] for key in keys], width, label="Reweighing", color="#318f82")
    ax.set_xticks(x[:3], names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Predictive performance on held-out test data")
    ax.legend(frameon=False)
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
) -> str:
    base = results["baseline"]
    mitigated = results["reweighing"]
    summary = (
        f"The held-out test set contained {results['test_n']} applicants. "
        f"Reweighing changed accuracy from {base['accuracy']:.3f} to "
        f"{mitigated['accuracy']:.3f} and F1 from {base['f1']:.3f} to "
        f"{mitigated['f1']:.3f}. Female-minus-male SPD changed from "
        f"{base['spd']:.3f} to {mitigated['spd']:.3f}; DI changed from "
        f"{base['di']:.3f} to {mitigated['di']:.3f}."
    )
    group_lines = []
    for _, row in group_table.iterrows():
        group_lines.append(
            f"| {row['model']} | {row['group']} | {int(row['n'])} | "
            f"{row['accuracy']:.3f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} |"
        )
    fairness_lines = []
    for name in ["spd", "di", "eod", "aod"]:
        fairness_lines.append(
            f"| {name.upper()} | {base[name]:.3f} | {mitigated[name]:.3f} | "
            f"{mitigated[name] - base[name]:+.3f} |"
        )
    ci_lines = []
    for _, row in bootstrap.iterrows():
        low, high = row["ci95_low"], row["ci95_high"]
        interval = f"[{low:+.3f}, {high:+.3f}]" if np.isfinite(low) else "not estimable"
        ci_lines.append(
            f"| {row['metric'].upper()} | {row['delta_mitigated_minus_baseline']:+.3f} | "
            f"{interval} | {'Yes' if row['ci_excludes_zero'] else 'No'} |"
        )

    female_rate = float(data.loc[data["sex"] == 1, "credit_good"].mean())
    male_rate = float(data.loc[data["sex"] == 0, "credit_good"].mean())
    missing_count = int(data.isna().sum().sum())
    base_men = group_table.loc[
        (group_table["model"] == "Baseline")
        & (group_table["group"] == "Men (privileged)")
    ].iloc[0]
    base_women = group_table.loc[
        (group_table["model"] == "Baseline")
        & (group_table["group"] == "Women (unprivileged)")
    ].iloc[0]
    mitigated_men = group_table.loc[
        (group_table["model"] == "AIF360 Reweighing")
        & (group_table["group"] == "Men (privileged)")
    ].iloc[0]
    mitigated_women = group_table.loc[
        (group_table["model"] == "AIF360 Reweighing")
        & (group_table["group"] == "Women (unprivileged)")
    ].iloc[0]
    female_n = int((data["sex"] == 1).sum())
    male_n = int((data["sex"] == 0).sum())
    eda_table = f"""| Summary | Value |
|---|---:|
| Applicants | {len(data)} |
| Model input features before encoding | {len([c for c in data.columns if c not in {"credit_class", "credit_good", "personal_status_sex"}])} |
| Missing cells | {missing_count} |
| Good-credit labels | {data["credit_good"].sum()} ({data["credit_good"].mean():.1%}) |
| Men / women | {male_n} / {female_n} |
| Median age | {data["age"].median():.1f} years |
| Median credit amount | {data["credit_amount"].median():,.0f} DM |"""
    group_interpretation = (
        f"On this test split, women had lower accuracy than men before mitigation "
        f"({base_women['accuracy']:.3f} vs. {base_men['accuracy']:.3f}) and after "
        f"mitigation ({mitigated_women['accuracy']:.3f} vs. {mitigated_men['accuracy']:.3f}). "
        f"The baseline female/male favorable prediction rates were "
        f"{base_women['predicted_good_rate']:.1%}/{base_men['predicted_good_rate']:.1%}; "
        f"after Reweighing they were {mitigated_women['predicted_good_rate']:.1%}/"
        f"{mitigated_men['predicted_good_rate']:.1%}. Thus women remained disadvantaged "
        f"in selection, though the gap narrowed. The baseline recall gap (women minus men) "
        f"was {base_women['recall'] - base_men['recall']:+.3f}; after mitigation it was "
        f"{mitigated_women['recall'] - mitigated_men['recall']:+.3f}. Women's accuracy "
        f"and recall improved in this split while men's declined, illustrating the "
        f"fairness-performance/group-error trade-off."
    )
    return f"""# Fairness Analysis of German Credit Classification Using IBM AIF360

## Abstract

This study examines group disparities in a logistic-regression classifier trained on the UCI Statlog German Credit data and tests AIF360 Reweighing as a training-data mitigation. Women are the unprivileged group and men the privileged group; good credit is the favorable label. The analysis reports predictive performance, group-wise metrics, Statistical Parity Difference (SPD), Disparate Impact (DI), Equal Opportunity Difference (EOD), and Average Odds Difference (AOD). {summary} These results describe one random split of a small historical benchmark and should not be read as evidence of deployment readiness.

## 1. Introduction

Credit decisions affect access to financial resources and can reproduce structural disadvantage when historical outcomes encode unequal access, institutional practices, or measurement bias. A model can have acceptable aggregate accuracy while exhibiting different selection rates or error rates across groups. Fairness assessment therefore requires both predictive and group-conditional metrics, together with a clear statement of which outcomes are favorable and how groups are defined.

This analysis uses the German Credit benchmark distributed by the UCI Machine Learning Repository (1,000 records, 20 input variables). The historical dataset is useful for method demonstration, not for estimating contemporary creditworthiness or the prevalence of discrimination. The socio-technical risk is especially salient because sex, age, marital status, employment, housing, and financial history may encode sensitive personal circumstances or act as proxies for unequal opportunity. Removing a protected field alone would not eliminate proxy information.

## 2. Methods

### 2.1 Data, outcome, and protected groups

The original target is coded 1 (good credit) or 2 (bad credit); it was mapped to favorable `credit_good = 1` and unfavorable `credit_good = 0`. Sex was derived from `personal_status_sex`: women (A92, A95) are unprivileged (`sex = 1`), and men (A91, A93, A94) are privileged (`sex = 0`). This operationalization follows the source variable's coding and does not imply that sex is binary in general. Across the complete data, the observed good-credit label rate was {male_rate:.1%} for men and {female_rate:.1%} for women. The file contained {missing_count} missing cells; preprocessing nonetheless includes median imputation for numeric variables and most-frequent imputation for categorical variables.

### 2.2 EDA and preprocessing

The pipeline records descriptive statistics, missingness, outcomes by sex, age-by-outcome distributions, and credit amount distributions.

{eda_table}

![Observed favorable label rate by sex](figures/eda_outcome_by_sex.png)

![Age distribution by credit class](figures/eda_age_by_outcome.png)

![Credit amount distribution](figures/eda_credit_amount.png)

It uses an 80/20 stratified train/test split (seed 42). Continuous features are median-imputed and standardized; categorical features are mode-imputed and one-hot encoded. The source `personal_status_sex` field is excluded after deriving sex, avoiding a redundant encoding; the derived sex feature is retained in the model so that the audit evaluates direct as well as proxy-related disparities. Preprocessing is fit on the training partition only.

### 2.3 Model and evaluation

Logistic regression was fit with a fixed 0.5 threshold. Accuracy, precision, recall, F1, and ROC-AUC are reported on the held-out test set. Group-wise estimates are computed separately for men and women. Small group sample sizes make some estimates uncertain; ROC-AUC is undefined if a group has only one outcome class.

### 2.4 AIF360 fairness metrics and mitigation

AIF360 `BinaryLabelDatasetMetric` computes SPD and DI; AIF360 `ClassificationMetric` computes EOD and AOD from true and predicted labels. All differences are unprivileged women minus privileged men. Thus SPD < 0 means women have a lower favorable prediction rate; EOD < 0 means lower true-positive rate; AOD < 0 indicates a net disadvantage across true- and false-positive rates. DI is women-to-men favorable selection rate, with 1 indicating parity. The 0.80 ratio is sometimes used as a screening heuristic; it is not a universal fairness definition or legal determination.

AIF360 `Reweighing` is fitted on training data only and assigns instance weights to reduce dependence between group membership and the outcome. A second logistic regression is trained using these weights. The test data are not reweighted. The same split, features, model family, and threshold are retained to make the comparison interpretable.

### 2.5 Uncertainty

A paired bootstrap stratified by sex and observed outcome (1,000 resamples, seed 42) estimates 95% intervals for selected test-set changes, defined as reweighing minus baseline. The interval-excludes-zero flag is an exploratory descriptive check, not a confirmatory hypothesis test; no multiple-comparison adjustment is applied.

## 3. Results

### 3.1 Overall performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline logistic regression | {base['accuracy']:.3f} | {base['precision']:.3f} | {base['recall']:.3f} | {base['f1']:.3f} | {base['roc_auc']:.3f} |
| AIF360 Reweighing | {mitigated['accuracy']:.3f} | {mitigated['precision']:.3f} | {mitigated['recall']:.3f} | {mitigated['f1']:.3f} | {mitigated['roc_auc']:.3f} |

![Predictive performance comparison](figures/performance_comparison.png)

### 3.2 Group-wise performance

| Model | Group | N | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(group_lines)}

Group differences can reflect both model behavior and unequal outcome prevalence. They should not be interpreted as causal effects of sex.

{group_interpretation}

### 3.3 Fairness measures and mitigation

| Measure | Baseline | Reweighing | Change |
|---|---:|---:|---:|
{chr(10).join(fairness_lines)}

![AIF360 fairness metric comparison](figures/fairness_comparison.png)

SPD and DI describe favorable prediction rates; EOD conditions on positive outcomes; AOD averages group differences in true- and false-positive rates. Values should be interpreted jointly: optimizing one criterion does not guarantee parity under the others. Both observed DI values are above the 0.80 screening heuristic, but this does not establish fairness or provide a legal conclusion.

### 3.4 Paired bootstrap changes

| Metric | Median change | 95% paired stratified bootstrap interval | Interval excludes zero |
|---|---:|---:|:---:|
{chr(10).join(ci_lines)}

Intervals that include zero do not establish that the methods are equivalent; they indicate that this sample and resampling procedure do not clearly resolve the direction of change. Small test-set subgroup counts and the single split constrain precision.

## 4. Discussion

Reweighing changes the contribution of training observations rather than removing the protected feature or imposing an output constraint. Its practical value must be judged by whether it improves a relevant fairness criterion without unacceptable loss in predictive utility. The results above make that trade-off explicit for this split. There is no single universally correct metric: statistical parity does not condition on qualification, while equal opportunity ignores false-positive disparity. AOD combines two error-rate differences and can conceal opposing component changes. DI can be unstable when the privileged selection rate is near zero. The measures do not capture intersectional harms, calibration, individual fairness, causal fairness, or the costs of errors to applicants and lenders.

There are additional dataset limitations. This is a small, historical, German benchmark with a binary sex coding and a constructed proxy outcome; its labels are not necessarily ground-truth creditworthiness. The UCI documentation also provides an asymmetric cost matrix: classifying a bad-credit applicant as good has a higher stated cost than the reverse. The reported conventional metrics weight false positives and false negatives symmetrically, so a consequential deployment study would need to justify an appropriate cost-sensitive evaluation with affected stakeholders. The feature distributions, institutional context, population, and legal framework differ from current lending. A single random split provides limited evidence, and measured associations cannot establish discrimination or causation. The experiment is an educational audit, not a recommendation to deploy a credit scoring system.

Deployment would require governance beyond model metrics: lawful purpose and data provenance; privacy and security controls; documented feature justification; meaningful human review and appeal; adverse-impact and consumer-protection review; accessibility; and clear accountability for overrides. Monitoring should track input and label drift, subgroup sample sizes, selection rates, calibration and error rates, confidence intervals, missingness, and intersectional slices. Metrics should be monitored by a versioned dashboard with predefined alert thresholds, review owners, and rollback criteria. Labels may arrive late and reflect prior decisions, so independent outcomes and ongoing stakeholder review are important. Any remediation should be tested for legal, operational, and stakeholder consequences before use.

## 5. Conclusion

This reproducible exercise evaluates a logistic-regression model with four AIF360 group fairness metrics and applies training-only Reweighing. The actual held-out outcomes and uncertainty estimates are reported above and exported as CSV/JSON for inspection. The analysis demonstrates how fairness conclusions depend on metric choice, group definition, threshold, and data context; it does not certify fairness or justify real credit decisions.

## Reproducibility and source

Run `python run_analysis.py` from the project root using the versions in `requirements.txt`. The script downloads and caches the German Credit file from the [UCI Machine Learning Repository, Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data). Citation: Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 (CC BY 4.0). The split seed is 42, test size is 20%, threshold is 0.5, and bootstrap count is 1,000. See `results.json`, CSV tables, and figures in this output directory for machine-readable results and diagnostics.
"""


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
    deck = Presentation()
    deck.slide_width = Inches(13.333)
    deck.slide_height = Inches(7.5)
    blank = deck.slide_layouts[6]

    slide = deck.slides.add_slide(blank)
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = NAVY
    add_text(slide, "FAIRNESS IN CREDIT SCORING", 0.8, 1.25, 11.6, 0.45, 15, TEAL, True)
    add_text(slide, "Measuring and mitigating group disparities", 0.8, 1.85, 11.7, 1.3, 34, RGBColor(255, 255, 255), True)
    add_text(slide, "German Credit benchmark • IBM AIF360 • 5-minute research summary", 0.84, 3.35, 11.4, 0.55, 17, RGBColor(218, 228, 238))
    add_text(slide, "Good credit is favorable; women are the unprivileged group in this audit.", 0.84, 5.55, 11.3, 0.6, 16, RGBColor(218, 228, 238))

    slide = deck.slides.add_slide(blank)
    add_slide_title(slide, "1 | Why audit this model?", "Context • about 40 seconds")
    add_bullet_list(
        slide,
        [
            "Credit access is consequential; historical data can encode structural inequality and proxy information.",
            "Dataset: 1,000 historical German credit applications and 20 source features.",
            "Outcome: class 1 = good credit (favorable); class 2 = bad credit.",
            "Women (A92, A95) are unprivileged; men (A91, A93, A94) are privileged.",
            "This is a method demonstration—not a current lending population or deployment-ready model.",
        ],
        0.9,
        1.75,
        11.7,
        4.9,
    )

    slide = deck.slides.add_slide(blank)
    add_slide_title(slide, "2 | Design and mitigation", "Methods • about 50 seconds")
    add_bullet_list(
        slide,
        [
            "80/20 stratified split (seed 42); preprocessing fitted on training data only.",
            "Median/mode imputation, scaled numeric data, one-hot categorical features.",
            "Logistic regression evaluated at threshold 0.50; sex retained to expose direct and proxy-related disparities.",
            "AIF360 Reweighing adjusts training-instance weights; same estimator and held-out test set.",
            "AIF360 metrics: SPD, DI, EOD, AOD. Differences are women minus men; DI is women-to-men selection rate.",
        ],
        0.9,
        1.75,
        11.7,
        4.9,
    )

    slide = deck.slides.add_slide(blank)
    add_slide_title(slide, "3 | Predictive performance", "Held-out test set • about 55 seconds")
    slide.shapes.add_picture(str(FIGURES / "performance_comparison.png"), Inches(0.75), Inches(1.6), width=Inches(7.2))
    base = results["baseline"]
    mitigated = results["reweighing"]
    add_text(
        slide,
        f"Accuracy\n{base['accuracy']:.3f} → {mitigated['accuracy']:.3f}\n\n"
        f"F1\n{base['f1']:.3f} → {mitigated['f1']:.3f}\n\n"
        f"ROC-AUC\n{base['roc_auc']:.3f} → {mitigated['roc_auc']:.3f}",
        8.45,
        1.85,
        4.0,
        3.7,
        size=22,
        color=NAVY,
        bold=True,
    )
    add_text(slide, "Baseline → Reweighing. Utility changes must be considered alongside group-specific errors.", 8.45, 5.85, 4.1, 0.75, size=13, color=MUTED)

    slide = deck.slides.add_slide(blank)
    add_slide_title(slide, "4 | Fairness metrics", "AIF360 • about 65 seconds")
    slide.shapes.add_picture(str(FIGURES / "fairness_comparison.png"), Inches(0.6), Inches(1.55), width=Inches(8.0))
    add_text(slide, "Baseline → Reweighing", 9.0, 1.65, 3.7, 0.35, 13, MUTED, True)
    for index, key in enumerate(["spd", "di", "eod", "aod"]):
        label = "DI" if key == "di" else key.upper()
        add_text(
            slide,
            f"{label}: {base[key]:.3f} → {mitigated[key]:.3f}",
            9.0,
            2.2 + index * 0.78,
            3.7,
            0.48,
            20,
            NAVY,
            True,
        )
    add_text(slide, "Parity: 0 for differences; DI = 1. A negative difference signals fewer favorable outcomes / lower TPR for women.", 0.85, 6.35, 11.7, 0.55, 13, MUTED)

    slide = deck.slides.add_slide(blank)
    add_slide_title(slide, "5 | Interpretation and next steps", "Takeaways • about 70 seconds")
    base_men = group_table.loc[
        (group_table["model"] == "Baseline")
        & (group_table["group"] == "Men (privileged)")
    ].iloc[0]
    base_women = group_table.loc[
        (group_table["model"] == "Baseline")
        & (group_table["group"] == "Women (unprivileged)")
    ].iloc[0]
    mitigated_men = group_table.loc[
        (group_table["model"] == "AIF360 Reweighing")
        & (group_table["group"] == "Men (privileged)")
    ].iloc[0]
    mitigated_women = group_table.loc[
        (group_table["model"] == "AIF360 Reweighing")
        & (group_table["group"] == "Women (unprivileged)")
    ].iloc[0]
    add_bullet_list(
        slide,
        [
            f"Women vs. men accuracy/recall: {base_women['accuracy']:.3f}/{base_women['recall']:.3f} baseline vs. {base_men['accuracy']:.3f}/{base_men['recall']:.3f}; after reweighing {mitigated_women['accuracy']:.3f}/{mitigated_women['recall']:.3f} vs. {mitigated_men['accuracy']:.3f}/{mitigated_men['recall']:.3f}.",
            "Reweighing targets training-set group/outcome association; it cannot repair biased labels or remove every proxy.",
            "Metric trade-offs are real: statistical parity, equal opportunity, and error-rate balance encode different goals.",
            "Bootstrap intervals quantify split-specific uncertainty; this one split is not a deployment guarantee.",
            "Before real use: legal and stakeholder review, appeals and human oversight, drift and intersectional monitoring, documented alerts and rollback.",
        ],
        0.9,
        1.75,
        11.7,
        4.75,
    )
    add_text(slide, "Conclusion: fairness is a continuing governance process, not a single score.", 0.9, 6.4, 11.5, 0.45, 17, TEAL, True)

    for slide_number, slide in enumerate(deck.slides, start=1):
        if slide_number == 1:
            continue
        add_text(slide, f"German Credit fairness analysis  •  {slide_number}/6", 9.2, 7.08, 3.4, 0.23, 9, MUTED, align=PP_ALIGN.RIGHT)
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

    baseline_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    baseline_model.fit(x_train, y_train)
    baseline_metrics, baseline_groups, baseline_pred, _ = evaluate_model(
        baseline_model, x_test, y_test, test["sex"]
    )

    train_dataset = make_dataset(x_train, y_train)
    reweigher = Reweighing(
        unprivileged_groups=UNPRIVILEGED,
        privileged_groups=PRIVILEGED,
    )
    reweighed_train = reweigher.fit_transform(train_dataset)
    mitigated_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    mitigated_model.fit(
        x_train,
        y_train,
        sample_weight=reweighed_train.instance_weights,
    )
    mitigated_metrics, mitigated_groups, mitigated_pred, _ = evaluate_model(
        mitigated_model, x_test, y_test, test["sex"]
    )

    group_table = pd.concat(
        [
            baseline_groups.assign(model="Baseline"),
            mitigated_groups.assign(model="AIF360 Reweighing"),
        ],
        ignore_index=True,
    )[["model", "group", "n", "observed_good_rate", "predicted_good_rate",
       "accuracy", "precision", "recall", "f1", "roc_auc"]]
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
        "baseline": baseline_metrics,
        "reweighing": mitigated_metrics,
    }
    metrics_table = pd.DataFrame(
        [dict(model="Baseline", **baseline_metrics),
         dict(model="AIF360 Reweighing", **mitigated_metrics)]
    )
    metrics_table.to_csv(OUTPUT / "metrics.csv", index=False)

    bootstrap = bootstrap_differences(
        y_test.to_numpy(dtype=int),
        test["sex"].to_numpy(dtype=int),
        baseline_pred,
        mitigated_pred,
    )
    bootstrap.to_csv(OUTPUT / "bootstrap_differences.csv", index=False)
    save_comparison_plot(baseline_metrics, mitigated_metrics)
    (OUTPUT / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    report = create_report(data, results, group_table, bootstrap)
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")
    create_presentation(results, group_table)

    print(f"Analysis complete. Outputs: {OUTPUT}")
    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nAIF360 fairness metric differences use women minus men.")
    print(bootstrap.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()

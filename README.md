# German Credit Fairness Analysis
**MSc-level project | Reproducible fairness analysis of the UCI German Credit dataset**

**Group Members:**
1. Betel Negusu
2. Esrom Adugna
3. Aklile Solomon

---

## Project Overview

This project provides a **reproducible, end-to-end fairness analysis** of the UCI German Credit dataset using **IBM AI Fairness 360 (AIF360)**. It is designed for MSc-level study and demonstrates how to:

- Download and explore the public German Credit dataset
- Perform data preprocessing and exploratory data analysis (EDA)
- Train logistic regression, random forest, and gradient boosting classifiers
- Apply AIF360 **Reweighing** mitigation technique
- Compute and compare fairness and performance metrics
- Visualize results via an interactive dashboard

The goal is to illustrate **direct and proxy-related disparities** with respect to the protected attribute **sex** (derived from the UCI `personal_status_sex` field), and to show how reweighing can adjust these disparities.

---

## Quick Start

> **Requirements:** Python 3.10–3.13

```powershell
# 1. Create and activate a virtual environment
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Run the analysis pipeline
python german_fairness.py
```

The pipeline:

1. **Downloads** the German Credit data (on first run) and caches it under `data/raw/`
2. **Performs EDA and preprocessing** (median/mode imputation, standardization, one-hot encoding)
3. **Trains three classifiers** with an 80/20 stratified train/test split and a fixed random seed (42)
4. **Applies AIF360 Reweighing** (fit on the training partition only) and retrains each classifier
5. **Writes metrics, figures, a report, and a PowerPoint presentation** to `outputs/`

---

## View Results in a Web Browser

After running the analysis, start the local Streamlit dashboard:

```powershell
python -m streamlit run app.py
```

The dashboard opens in your default browser (normally at `http://localhost:8501`). It includes:

- **Performance metrics** (accuracy, precision, recall, F1)
- **Fairness metrics** (SPD, EOD, AOD, DI) with group-wise comparisons
- **Bootstrap confidence intervals** (1,000 resamples) for mitigated-minus-baseline differences
- **EDA plots** (distributions, correlation matrix, etc.)
- **Six model comparisons** across baseline and reweighed logistic regression, random forest, and gradient boosting
- **Full research report** (REPORT.md)

Use **Re-run analysis** in the sidebar to refresh results. The dashboard runs locally — no project data is uploaded.

---

## Study Design

| Component | Details |
|---|---|
| **Outcome** | Good credit (UCI class 1) is favorable; bad credit (class 2) is unfavorable |
| **Protected attribute** | Sex: Women (A92, A95) are the unprivileged group (`sex = 1`); men (A91, A93, A94) are the privileged group (`sex = 0`) |
| **Models** | Logistic regression (primary model), random forest, and gradient boosting. Median/mode imputation, numeric standardization, and one-hot encoding of categorical features. The derived sex feature is retained for audit. |
| **Mitigation** | AIF360 `Reweighing` is fit on the training partition only; its instance weights are supplied to each estimator. |
| **Fairness signs** | SPD, EOD, and AOD are computed as women minus men. Zero = parity; negative values indicate women receive fewer favorable outcomes or have lower error-rate performance. DI is the women-to-men favorable prediction rate ratio; 1 = parity. The 0.80 rule is reported only as a screening heuristic, not as a legal conclusion. |

---

## Outputs

Running `python german_fairness.py` creates the following under `outputs/`:

| File / Directory | Description |
|---|---|
| `REPORT.md` | Fifteen-page report covering the workflow, methods, results, limitations, and reproducibility. A copy is also maintained in `documentation/REPORT.md`. |
| `presentation.pptx` | Twelve-slide presentation covering the project, methods, results, limitations, and reproducibility. |
| `metrics.csv` | Aggregate held-out performance and fairness metrics for all model/method combinations. |
| `group_metrics.csv` | Fairness and performance metrics broken down by protected group. |
| `bootstrap_differences.csv` | Bootstrap resample differences (mitigated minus baseline). |
| `figures/` | EDA and fairness comparison charts (PNG files). |
| `results.json` | Full-precision machine-readable results. |

> **Important:** Results concern this specific benchmark and data split only. They are **not** evidence that the model is appropriate for real lending decisions.

---

## License

This project is open source and available under the MIT License. Feel free to adapt the pipeline for other tabular datasets and fairness interventions.

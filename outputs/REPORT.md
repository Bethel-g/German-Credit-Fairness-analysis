# Fairness Analysis of German Credit Classification Using IBM AIF360

## Abstract

This study examines group disparities in a logistic-regression classifier trained on the UCI Statlog German Credit data and tests AIF360 Reweighing as a training-data mitigation. Women are the unprivileged group and men the privileged group; good credit is the favorable label. The analysis reports predictive performance, group-wise metrics, Statistical Parity Difference (SPD), Disparate Impact (DI), Equal Opportunity Difference (EOD), and Average Odds Difference (AOD). The held-out test set contained 200 applicants. Reweighing changed accuracy from 0.705 to 0.705 and F1 from 0.793 to 0.792. Female-minus-male SPD changed from -0.083 to -0.021; DI changed from 0.889 to 0.970. These results describe one random split of a small historical benchmark and should not be read as evidence of deployment readiness.

## 1. Introduction

Credit decisions affect access to financial resources and can reproduce structural disadvantage when historical outcomes encode unequal access, institutional practices, or measurement bias. A model can have acceptable aggregate accuracy while exhibiting different selection rates or error rates across groups. Fairness assessment therefore requires both predictive and group-conditional metrics, together with a clear statement of which outcomes are favorable and how groups are defined.

This analysis uses the German Credit benchmark distributed by the UCI Machine Learning Repository (1,000 records, 20 input variables). The historical dataset is useful for method demonstration, not for estimating contemporary creditworthiness or the prevalence of discrimination. The socio-technical risk is especially salient because sex, age, marital status, employment, housing, and financial history may encode sensitive personal circumstances or act as proxies for unequal opportunity. Removing a protected field alone would not eliminate proxy information.

## 2. Methods

### 2.1 Data, outcome, and protected groups

The original target is coded 1 (good credit) or 2 (bad credit); it was mapped to favorable `credit_good = 1` and unfavorable `credit_good = 0`. Sex was derived from `personal_status_sex`: women (A92, A95) are unprivileged (`sex = 1`), and men (A91, A93, A94) are privileged (`sex = 0`). This operationalization follows the source variable's coding and does not imply that sex is binary in general. Across the complete data, the observed good-credit label rate was 72.3% for men and 64.8% for women. The file contained 0 missing cells; preprocessing nonetheless includes median imputation for numeric variables and most-frequent imputation for categorical variables.

### 2.2 EDA and preprocessing

The pipeline records descriptive statistics, missingness, outcomes by sex, age-by-outcome distributions, and credit amount distributions.

| Summary | Value |
|---|---:|
| Applicants | 1000 |
| Model input features before encoding | 20 |
| Missing cells | 0 |
| Good-credit labels | 700 (70.0%) |
| Men / women | 690 / 310 |
| Median age | 33.0 years |
| Median credit amount | 2,320 DM |

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
| Baseline logistic regression | 0.705 | 0.779 | 0.807 | 0.793 | 0.754 |
| AIF360 Reweighing | 0.705 | 0.783 | 0.800 | 0.792 | 0.752 |

![Predictive performance comparison](figures/performance_comparison.png)

### 3.2 Group-wise performance

| Model | Group | N | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Men (privileged) | 140 | 0.736 | 0.800 | 0.840 | 0.820 | 0.784 |
| Baseline | Women (unprivileged) | 60 | 0.633 | 0.725 | 0.725 | 0.725 | 0.680 |
| AIF360 Reweighing | Men (privileged) | 140 | 0.721 | 0.802 | 0.810 | 0.806 | 0.785 |
| AIF360 Reweighing | Women (unprivileged) | 60 | 0.667 | 0.738 | 0.775 | 0.756 | 0.679 |

Group differences can reflect both model behavior and unequal outcome prevalence. They should not be interpreted as causal effects of sex.

On this test split, women had lower accuracy than men before mitigation (0.633 vs. 0.736) and after mitigation (0.667 vs. 0.721). The baseline female/male favorable prediction rates were 66.7%/75.0%; after Reweighing they were 70.0%/72.1%. Thus women remained disadvantaged in selection, though the gap narrowed. The baseline recall gap (women minus men) was -0.115; after mitigation it was -0.035. Women's accuracy and recall improved in this split while men's declined, illustrating the fairness-performance/group-error trade-off.

### 3.3 Fairness measures and mitigation

| Measure | Baseline | Reweighing | Change |
|---|---:|---:|---:|
| SPD | -0.083 | -0.021 | +0.062 |
| DI | 0.889 | 0.970 | +0.081 |
| EOD | -0.115 | -0.035 | +0.080 |
| AOD | -0.045 | 0.008 | +0.052 |

![AIF360 fairness metric comparison](figures/fairness_comparison.png)

SPD and DI describe favorable prediction rates; EOD conditions on positive outcomes; AOD averages group differences in true- and false-positive rates. Values should be interpreted jointly: optimizing one criterion does not guarantee parity under the others. Both observed DI values are above the 0.80 screening heuristic, but this does not establish fairness or provide a legal conclusion.

### 3.4 Paired bootstrap changes

| Metric | Median change | 95% paired stratified bootstrap interval | Interval excludes zero |
|---|---:|---:|:---:|
| ACCURACY | +0.000 | [-0.025, +0.020] | No |
| F1 | -0.001 | [-0.021, +0.017] | No |
| SPD | +0.062 | [+0.017, +0.119] | Yes |
| DI | +0.079 | [+0.022, +0.166] | Yes |
| EOD | +0.080 | [+0.020, +0.165] | Yes |
| AOD | +0.052 | [+0.015, +0.100] | Yes |

Intervals that include zero do not establish that the methods are equivalent; they indicate that this sample and resampling procedure do not clearly resolve the direction of change. Small test-set subgroup counts and the single split constrain precision.

## 4. Discussion

Reweighing changes the contribution of training observations rather than removing the protected feature or imposing an output constraint. Its practical value must be judged by whether it improves a relevant fairness criterion without unacceptable loss in predictive utility. The results above make that trade-off explicit for this split. There is no single universally correct metric: statistical parity does not condition on qualification, while equal opportunity ignores false-positive disparity. AOD combines two error-rate differences and can conceal opposing component changes. DI can be unstable when the privileged selection rate is near zero. The measures do not capture intersectional harms, calibration, individual fairness, causal fairness, or the costs of errors to applicants and lenders.

There are additional dataset limitations. This is a small, historical, German benchmark with a binary sex coding and a constructed proxy outcome; its labels are not necessarily ground-truth creditworthiness. The UCI documentation also provides an asymmetric cost matrix: classifying a bad-credit applicant as good has a higher stated cost than the reverse. The reported conventional metrics weight false positives and false negatives symmetrically, so a consequential deployment study would need to justify an appropriate cost-sensitive evaluation with affected stakeholders. The feature distributions, institutional context, population, and legal framework differ from current lending. A single random split provides limited evidence, and measured associations cannot establish discrimination or causation. The experiment is an educational audit, not a recommendation to deploy a credit scoring system.

Deployment would require governance beyond model metrics: lawful purpose and data provenance; privacy and security controls; documented feature justification; meaningful human review and appeal; adverse-impact and consumer-protection review; accessibility; and clear accountability for overrides. Monitoring should track input and label drift, subgroup sample sizes, selection rates, calibration and error rates, confidence intervals, missingness, and intersectional slices. Metrics should be monitored by a versioned dashboard with predefined alert thresholds, review owners, and rollback criteria. Labels may arrive late and reflect prior decisions, so independent outcomes and ongoing stakeholder review are important. Any remediation should be tested for legal, operational, and stakeholder consequences before use.

## 5. Conclusion

This reproducible exercise evaluates a logistic-regression model with four AIF360 group fairness metrics and applies training-only Reweighing. The actual held-out outcomes and uncertainty estimates are reported above and exported as CSV/JSON for inspection. The analysis demonstrates how fairness conclusions depend on metric choice, group definition, threshold, and data context; it does not certify fairness or justify real credit decisions.

## Reproducibility and source

Run `python run_analysis.py` from the project root using the versions in `requirements.txt`. The script downloads and caches the German Credit file from the [UCI Machine Learning Repository, Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data). Citation: Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 (CC BY 4.0). The split seed is 42, test size is 20%, threshold is 0.5, and bootstrap count is 1,000. See `results.json`, CSV tables, and figures in this output directory for machine-readable results and diagnostics.

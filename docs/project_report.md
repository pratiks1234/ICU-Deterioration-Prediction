
# Project Report: ICU Hospital Mortality Risk Prediction

## 1. Project Objective

This project develops an explainable machine learning pipeline for estimating hospital mortality risk using information available during the first 24 hours of an ICU stay.

The completed MVP predicts:

* `0`: patient survived the hospital admission
* `1`: patient died during the hospital admission

The current system predicts hospital mortality and should not yet be described as a real-time deterioration model or as a model predicting deterioration within a future 6-, 12-, or 24-hour window.

## 2. Dataset

The project uses the MIMIC-IV demo dataset.

The final cohort contains approximately:

* 100 unique patients
* 140 ICU stays
* 20 hospital mortality cases

Raw and processed patient-level data are not included in the public repository.

## 3. Cohort Construction

Patient, hospital-admission, and ICU-stay tables were merged using:

* `subject_id`
* `hadm_id`
* `stay_id`

Each row in the modeling dataset represents one ICU stay.

The target variable is:

```text
hospital_expire_flag
```

## 4. Observation Window

Vitals and laboratory measurements were extracted from the first 24 hours following ICU admission.

Only measurements satisfying the following condition were used:

```text
0 hours <= measurement time - ICU admission time <= 24 hours
```

This ensures that the model uses early ICU information rather than measurements recorded later during hospitalization.

## 5. Feature Engineering

### Admission and patient features

The baseline model used:

* Gender
* Age
* Admission type
* Admission location
* Insurance
* Race
* First ICU care unit

### Vital-sign features

The pipeline extracted:

* Heart rate
* Respiratory rate
* Oxygen saturation
* Systolic blood pressure
* Diastolic blood pressure
* Mean arterial pressure
* Temperature

For each vital sign, summary statistics such as mean, minimum, maximum, standard deviation, count, and latest value were created.

### Laboratory features

The laboratory pipeline included:

* Lactate
* Creatinine
* Blood urea nitrogen
* Glucose
* Sodium
* Potassium
* Chloride
* Bicarbonate
* Hemoglobin
* Platelets
* White blood cell count
* INR
* PT
* PTT

Summary statistics were generated over the first 24-hour observation window.

## 6. Models Evaluated

Four XGBoost feature configurations were compared:

1. Admission-information baseline
2. Vitals-only model
3. Vitals-and-labs model
4. Reduced clinical model

The full vitals-and-labs model contained 154 input features. Because the demo cohort contains only 140 ICU stays, this model was vulnerable to overfitting.

A reduced clinical model was therefore created using 16 selected demographic, ICU, vital-sign, and laboratory features.

## 7. Reduced Clinical Feature Set

The reduced model used:

* Gender
* Age
* Admission type
* First ICU unit
* Mean heart rate
* Maximum heart rate
* Mean respiratory rate
* Minimum oxygen saturation
* Minimum mean arterial pressure
* Maximum temperature
* Maximum lactate
* Latest creatinine
* Latest blood urea nitrogen
* Maximum white blood cell count
* Minimum platelet count
* Minimum bicarbonate

## 8. Cross-Validation Results

Repeated five-fold stratified cross-validation was used to compare model configurations.

| Model                  | Feature Count |  AUROC |  AUPRC | Recall |     F1 |
| ---------------------- | ------------: | -----: | -----: | -----: | -----: |
| Admission baseline     |             7 | 0.7938 | 0.5322 |   0.50 | 0.4339 |
| Vitals only            |            56 | 0.6675 | 0.3601 |   0.12 | 0.1434 |
| Vitals and labs        |           154 | 0.7458 | 0.4603 |   0.24 | 0.2897 |
| Reduced clinical model |            16 | 0.7833 | 0.5357 |   0.39 | 0.3753 |

The reduced clinical model achieved:

```text
AUROC: 0.7833 ± 0.0998
AUPRC: 0.5357 ± 0.1815
```

The reduced model achieved the highest average AUPRC while using substantially fewer features than the full clinical model.

These results suggest that reducing the feature space helped limit overfitting on the small demo cohort.

## 9. Out-of-Fold Evaluation

Five-fold out-of-fold predictions were generated.

For each ICU stay:

1. The model was trained on four folds.
2. The selected ICU stay belonged to the remaining fold.
3. The prediction was generated without training on that ICU stay.

The out-of-fold results were:

```text
AUROC: 0.7767
AUPRC: 0.4809
```

At the default threshold of `0.50`, the confusion matrix was:

```text
[[108, 12],
 [ 13,  7]]
```

The model identified 7 of 20 hospital mortality cases at this threshold.

## 10. Threshold Selection

Thresholds from `0.05` to `0.95` were evaluated using the out-of-fold probabilities.

The threshold with the highest F1-score was:

```text
0.38
```

Performance at this threshold:

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          |  0.800 |
| Balanced accuracy |  0.717 |
| Precision         |  0.375 |
| Recall            |  0.600 |
| Specificity       |  0.833 |
| F1-score          |  0.462 |
| True positives    |     12 |
| False negatives   |      8 |
| False positives   |     20 |
| True negatives    |    100 |

The tuned threshold detected 12 of 20 mortality cases, compared with 7 cases at the default threshold.

This improvement came with an increase in false-positive warnings.

## 11. Risk Categories

The demonstration dashboard uses:

```text
Risk below 0.21       -> Low Risk
Risk from 0.21 to 0.38 -> Medium Risk
Risk of 0.38 or above -> High Risk
```

These categories are exploratory and have not been clinically validated.

## 12. Explainability

SHAP is used to explain individual model predictions.

Positive SHAP values move the model output toward higher predicted mortality risk, while negative SHAP values move the model output toward lower predicted risk.

SHAP explains model behavior. It does not prove that a feature causes mortality or that changing the feature would change the patient outcome.

## 13. Streamlit Dashboard

The dashboard displays:

* Out-of-fold mortality risk
* Risk category
* Actual hospital outcome
* Final-model risk
* Patient and ICU information
* First-24-hour vital summaries
* First-24-hour laboratory summaries
* Risk-increasing SHAP features
* Risk-decreasing SHAP features

The dashboard is intended only for research, education, and portfolio demonstration.

## 14. Main Findings

The project produced several important findings:

1. A simple admission-information model performed strongly on the small demo cohort.
2. Adding all available vitals and labs increased model complexity and reduced generalization.
3. A reduced clinical feature set recovered much of the predictive performance while remaining clinically interpretable.
4. Out-of-fold predictions provided a more honest patient-level evaluation than predictions from a model trained on the complete dataset.
5. Threshold selection substantially changed the balance between mortality detection and false-positive warnings.

## 15. Limitations

The project has several major limitations:

* The MIMIC-IV demo dataset is extremely small.
* Only approximately 20 mortality cases are available.
* Performance varies substantially across validation folds.
* The current target is hospital mortality, not future deterioration.
* Cross-validation currently separates ICU stays rather than grouping all stays from the same patient.
* Different ICU stays from one patient may therefore appear in different folds.
* The selected threshold is exploratory.
* The model has not been externally validated.
* The model probabilities have not been calibrated.
* Some features may reflect care processes rather than only patient physiology.
* Feature importance and SHAP values are not causal.
* The dashboard is not suitable for clinical decision-making.

## 16. Ethical Considerations

Healthcare machine learning systems can produce different error rates across demographic groups.

Variables such as race and insurance may capture structural and social differences rather than biological risk. These variables should be carefully evaluated and may be more appropriate for fairness auditing than for final clinical prediction.

A production-quality system would require:

* Subgroup performance analysis
* Fairness evaluation
* Probability calibration
* External validation
* Prospective validation
* Clinical workflow testing
* Governance and monitoring

## 17. Future Work

Future improvements include:

* Using `StratifiedGroupKFold` with `subject_id`
* Training on the full credentialed MIMIC-IV dataset
* Defining future deterioration outcomes
* Creating 6-, 12-, and 24-hour prediction horizons
* Building longitudinal time-series features
* Evaluating LSTM and GRU models
* Performing probability calibration
* Conducting fairness analysis
* Adding external validation
* Deploying a reproducible demonstration application

## 18. Conclusion

This project demonstrates an end-to-end healthcare machine learning workflow including cohort construction, clinical feature engineering, model comparison, class-imbalance handling, out-of-fold evaluation, threshold tuning, SHAP explainability, and dashboard development.

The results validate the technical pipeline, but they should not be interpreted as evidence of clinical readiness because of the small demo dataset and lack of external validation.

## Disclaimer

This project is for research, education, and portfolio demonstration only. It is not a medical device and must not be used for diagnosis, treatment, patient monitoring, or clinical decision support.

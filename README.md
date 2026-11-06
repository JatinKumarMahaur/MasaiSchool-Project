# How to run

* pip install -r requirements.txt
* python generate_data.py          # prints: 24 1224 48
* python part_a_clean_eda.py
* python part_b_regression.py
* python part_c_classification.py
* python part_d_segments.py
* python package_model.py
* python predict_delay.py

It is compulsory to run the scripts must be run in this order.

# Data findings

* Data cleaning: The original 1,224 records were reduced to 1,200 shipments after removing 24 duplicates.
* Overall delays: 278 shipments were delayed, giving a delay rate of 23.17%.
* Service type: Express shipments had the highest delay rate (48.97%), followed by Standard (20.56%) and Economy (8.57%). Express has a shorter delivery time limit, making delays more likely.
* Hub tier: Higher-tier hubs had more delays. Tier 1 had 4%, Tier 2 had 21.5%, and Tier 3 had 44%.
* Distance: Longer distances led to more delays. Shipments over 601 km had the highest delay rate of 85.23%.
* Main factors: Promised delivery hours, distance and number of stops were strongly linked to delivery time. Hubs with higher daily capacity generally delivered faster.
* Log transformation: The declared parcel value was converted using log because a few very expensive parcels made the data uneven. After conversion, the values became more balanced.
* Missing data: Missing values, incorrect weights and inconsistent service names were corrected. After cleaning, all 1,200 records had complete data.

<b>Overall conclusion:</b> Shipment delays are mainly associated with longer distances, higher hub tiers, express service and more delivery stops. Improving hub performance and planning long-distance deliveries could help reduce delays.

# Model results (regression)

## 1. Data Splitting

The dataset was divided into three parts:

- **Training data:** 840 rows
- **Validation data:** 180 rows
- **Testing data:** 180 rows

The scaler was fitted only on the training data to avoid data leakage and ensure fair model evaluation.

## 2. Baseline Model

The baseline model achieved an R² of **-0.0212** and an RMSE of **18.8552 hours**.

This means the baseline model could not predict delivery times accurately.

## 3. Model Comparison

Linear Regression performed best on the validation data, with an RMSE of 4.8912 hours and an R² of 0.9313.

| Model | Validation RMSE | R² |
|---|---:|---:|
| Linear Regression | 4.8912 | 0.9313 |
| Gradient Boosting | 4.9165 | 0.9306 |
| Random Forest | 5.5095 | 0.9128 |
| Decision Tree | 6.2796 | 0.8867 |

## 4. Bias and Variance

- Shallow trees (depth 1 and 2) had high bias and could not learn the data properly.
- Very deep trees had high variance and overfitted the training data.
- Depth 6 gave the best validation result for the Decision Tree.

## 5. Learning Rate

For Gradient Boosting, a learning rate of **0.1** performed best, with an R² of 0.9306.

- **0.01:** Too slow, resulting in underfitting.
- **0.1:** Best performance.
- **1.0:** Too high, causing the model to learn noise.

## 6. Cross-Validation

Five-fold cross-validation gave the following results:

| Model | Mean R² | Standard Deviation |
|---|---:|---:|
| Gradient Boosting | 0.9353 | 0.0045 |
| Linear Regression | 0.9261 | 0.0048 |

Gradient Boosting performed slightly better across different data splits. It can also capture complex patterns, such as extra delays during sales periods and rainy weather.

## 7. Final Test Results

Based on the rule of selecting the model with the lowest validation RMSE, Linear Regression was selected for final testing.

- **RMSE:** 4.5235 hours
- **MAE:** 3.5671 hours
- **R²:** 0.9493

Gradient Boosting was not tested again, keeping the test data untouched during model selection.

# Model Results (Classification)

## 1. Data Splitting

The dataset was divided into three parts using stratified splitting:

- **Training data:** 840 rows
- **Validation data:** 180 rows
- **Testing data:** 180 rows

The delayed shipments were distributed as follows:

- Training: 195 delayed parcels
- Validation: 42 delayed parcels
- Testing: 41 delayed parcels

## 2. Baseline Model

The baseline model predicted every shipment as "Not Delayed".

- **Accuracy:** 76.67%
- **True Negatives (TN):** 138
- **False Negatives (FN):** 42
- **Recall:** 0%

Although accuracy looked good, the model failed to identify any delayed shipments. Therefore, it was not useful for the business goal.

## 3. Model Comparison

The trained models achieved accuracy between 88.33% and 90.56%.

| Model | F1 Score |
|---|---:|
| Logistic Regression | 0.8046 |
| Decision Tree (Depth 6) | 0.7470 |
| Random Forest | 0.7848 |

Logistic Regression performed best among these models based on F1 score.

## 4. Feature Selection and Noise Columns

The five bookkeeping columns had no useful information but contributed 22.44% of the Random Forest's feature importance.

The seven most useful features were:

1. distance_km
2. hub_tier
3. hub_daily_capacity
4. n_stops
5. promised_hours
6. rain_flag
7. sale_event_flag

The bookkeeping columns had very low F-scores, so they were removed to improve the model.

## 5. Selecting the Best Number of Features (K)

Different numbers of features were tested.

| Number of Features (K) | Validation F1 |
|---|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

**K = 7** gave the best result.

Using only five features removed some useful information, while adding more features introduced unnecessary noise.

## 6. Model Tuning

Grid Search and Randomized Search were used to improve the Random Forest model.

| Method | Validation F1 |
|---|---:|
| Grid Search | 0.7750 |
| Randomized Search | 0.7848 |
| Default Random Forest | 0.8333 |

The default Random Forest performed better than both tuning methods.

Therefore, the final model uses:
- 100 trees
- No maximum depth limit
- Minimum samples per leaf = 1
- Top 7 selected features

This shows that selecting good features can be more useful than unnecessary model tuning.

## 7. Alert Threshold Selection

The business requirement was to achieve at least **80% recall** on validation data.

| Threshold | Precision | Recall |
|---|---:|---:|
| 0.6 | — | 0.7619 |
| 0.5 | 0.7955 | 0.8333 |
| 0.4 | 0.7400 | 0.8810 |
| 0.1 | 0.4824 | — |

A threshold of **0.5** was selected because it was the highest threshold that achieved the required 80% recall.

At this threshold:
- True Positives (TP): 35
- False Positives (FP): 9
- False Negatives (FN): 7

## 8. Final Test Results

The final Random Forest model was tested once using a threshold of 0.5.

| Metric | Test Result |
|---|---:|
| Accuracy | 86.67% |
| Precision | 68.89% |
| Recall | 75.61% |
| F1 Score | 0.7209 |

Confusion matrix:

- True Negatives (TN): 125
- False Positives (FP): 14
- False Negatives (FN): 10
- True Positives (TP): 31

The test recall was 75.61%, which is lower than the validation recall of 83.33% and below the target of 80%.

This difference may partly be due to the small test sample, which contained only 41 delayed shipments.

# Operations Recommendation

## 1. Hub Segmentation

K-Means clustering divided the hubs into three groups based on their delivery performance.

The three segments are:

- **Escalate:** 9 hubs with a delay rate of 46%.
- **Watch:** 8 hubs with a delay rate of 15.5%.
- **Standard:** 7 hubs with a delay rate of 2.57%.

The Escalate segment has the highest delays, longer delivery distances and more delivery stops. These hubs need extra attention from management.

### Escalate Hubs

The following 9 hubs belong to the Escalate category:

- Nagpur (46% delay rate)
- Kochi (36%)
- Patna (46%)
- Ranchi (38%)
- Guwahati (62%)
- Raipur (48%)
- Bhubaneswar (40%)
- Siliguri (44%)
- Agartala (54%)

## 2. Hub Tier vs Actual Performance

Some hubs perform differently from what their assigned tier suggests.

- **Kolkata (Tier 1):** Behaves like a Watch hub, so its performance is weaker than expected.
- **Nagpur (Tier 2):** Behaves like an Escalate hub and has higher delay risk.
- **Kochi (Tier 2):** Also behaves like an Escalate hub.
- **Dehradun (Tier 3):** Behaves like a Watch hub, performing better than its tier suggests.

Therefore, operational decisions should be based on actual hub performance rather than only tier labels.

## 3. Alert Threshold

A shipment should be flagged when its predicted delay probability is **50% or higher**.

Test results show:

- **Precision:** 68.89% – Around 7 out of 10 flagged shipments are actually delayed.
- **Recall:** 75.61% – Around 8 out of 10 delayed shipments are identified.

Validation recall was 83.33%. Therefore, operations should expect the model to identify roughly 75–83% of delayed shipments.

## 4. Actions for New Shipments

Out of 48 new shipments:

- **8 shipments** were flagged by the model.
- **6 shipments** belong to Escalate hubs and need attention from hub managers.
- **2 shipments** need proactive customer calls.
- **40 shipments** do not require any immediate action.

This helps the operations team focus on shipments that have a higher risk of delay.

## 5. Why K = 3 Was Selected

Different cluster values were tested to find the best number of groups.

| K Value | Across-to-Within Ratio |
|---|---:|
| 2 | 2.8940 |
| 3 | 5.3928 |
| 4 | 6.3047 |
| 5 | 6.8866 |

K = 3 was selected because:

- It showed the biggest improvement in cluster separation.
- It created three clearly different groups.
- Each group had enough hubs to manage.
- The business already follows three operational playbooks: Standard, Watch and Escalate.

Higher values of K created smaller groups that would be difficult to manage separately.

## 6. Overall Operations Recommendation

The company should use the three hub segments to improve delivery performance.

- **Escalate hubs:** Involve hub managers and take immediate action to reduce delays.
- **Watch hubs:** Monitor shipments regularly and take preventive action.
- **Standard hubs:** Continue normal operations with regular monitoring.

The company should also use the 50% alert threshold to identify risky shipments early and contact customers when needed.

# Production Plan

## 1. Production Readiness

The project is ready for production based on the following points:

### Accuracy
Model performance is checked using validation and test results stored in `part_b_metrics.csv` and `part_c_final_test.csv`. The results are also compared with the baseline model.

### Reliability
The `clean_shipments` function handles:
- Duplicate records
- Inconsistent text formats and casing
- Missing values
- Incorrect weight values (999 kg)
- Unknown payment modes

The `predict_delay.py` script also checks that no feature values are missing.

### Reproducibility
The project uses:
- Fixed library versions in `requirements.txt`
- `generate_data.py` to create the dataset
- Fixed `random_state=42`
- A defined execution order in the "How to Run" section.

This helps produce consistent results.

### Performance (Speed)
The model predicts delays for multiple shipments in a single batch using `predict_proba`. The Random Forest model uses only seven selected features, making it fast enough for on-demand predictions.

### Maintainability
The project is easy to manage because:
- `src/shipment_prep.py` contains shared preprocessing functions.
- Each project part has a separate script.
- Important constants such as `FEATURES` and `PROMISED_HOURS` are defined in one place.

### Consistency
Training and prediction use the same preprocessing and feature-building functions. Stored median values and feature order are maintained in:

- `models/impute_values.csv`
- `models/feature_order.csv`

This ensures consistent predictions.

### Monitoring
The file `new_shipment_actions.csv` stores the following information for every shipment:

- Delay probability
- Delay alert flag
- Hub segment
- Recommended action

These predictions can later be compared with actual delivery results.

## 2. Important Metrics to Monitor

The following metrics should be checked regularly after deployment.

| Metric | Current Value | Alert Condition |
|---|---:|---|
| Weekly delay rate | 23.17% | Outside 17%–29% for two consecutive weeks |
| Weekly flagged shipments | 25% | Outside 15%–35% |
| Test recall | 75.61% | Below 65% |
| Missing distance values | 3% | Above 10% in a week |
| Missing parcel weight values | 2% | Monitor for sudden increases |

### A. Weekly Delay Rate

The original training delay rate was 23.17%.

The company should investigate if the weekly delay rate goes below 17% or above 29% for two consecutive weeks.

### B. Flagged Shipment Rate

In the test data, 45 out of 180 shipments were flagged, representing 25%.

The company should investigate if the weekly flagged shipment rate goes below 15% or above 35%.

### C. Model Recall

The test recall was 75.61%.

Recall should be monitored over a rolling four-week period, with at least 40 delayed shipments whose actual outcomes are known.

If recall falls below 65%, the model should be reviewed and retrained if necessary.

### D. Missing Values

Missing values in distance and parcel weight should be monitored regularly.

In the training data:
- Distance was missing in 36 out of 1,200 shipments (3%).
- Parcel weight was missing in 24 out of 1,200 shipments (2%).

An alert should be raised if missing distance values exceed 10% in a week because too many values would need to be filled using the median.

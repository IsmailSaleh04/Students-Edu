# Student Performance Analytics

**Pipeline README & Technical Documentation**
`edu_exam.sqlite` · Binary Classification & Regression · v1.0

---

## Table of Contents

1. [Database Overview](#1-database-overview)
   - [1.1 Table Inventory](#11-table-inventory)
   - [1.2 Entity Relationship Overview](#12-entity-relationship-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
   - [2.1 Shared Steps](#21-shared-steps)
   - [2.2 Goal-Specific Steps](#22-goal-specific-steps)
3. [Feature Engineering](#3-feature-engineering)
   - [3.1 The Bridge Join](#31-the-bridge-join)
   - [3.2 Feature Groups](#32-feature-groups)
4. [Goal 1 — Grade Prediction (Regression)](#4-goal-1--grade-prediction-regression)
   - [4.1 Problem Definition](#41-problem-definition)
   - [4.2 Target Distribution](#42-target-distribution)
   - [4.3 EDA Highlights](#43-eda-highlights)
   - [4.4 Preprocessing](#44-preprocessing)
   - [4.5 Models Evaluated](#45-models-evaluated)
   - [4.6 Evaluation Metrics](#46-evaluation-metrics--goal-1)
   - [4.7 Key Observations](#47-key-observations--goal-1)
5. [Goal 2 — Failure Risk Classification](#5-goal-2--failure-risk-classification)
   - [5.1 Problem Definition](#51-problem-definition)
   - [5.2 Target Distribution](#52-target-distribution)
   - [5.3 EDA Highlights](#53-eda-highlights)
   - [5.4 Models Evaluated](#54-models-evaluated)
   - [5.5 Evaluation Metrics](#55-evaluation-metrics--goal-2)
   - [5.6 Model Diagnostics](#56-model-diagnostics)
   - [5.7 Feature Importance & Explainability](#57-feature-importance--explainability)
   - [5.8 Key Observations](#58-key-observations--goal-2)
6. [Limitations & Future Work](#6-limitations--future-work)
7. [Repository Structure](#7-repository-structure)
8. [Dependencies](#8-dependencies)

---

## 1. Database Overview

The project uses `edu_exam.sqlite`, a relational SQLite database with 16 tables capturing every dimension of student academic life — from raw attendance records to exam integrity flags and LMS interaction logs.

### 1.1 Table Inventory

| Table | Rows (approx.) | Description |
|---|---|---|
| `semesters` | 4 | Semester metadata (Fall/Spring, year) |
| `students` | 220 | Student profile — motivation, stress, year of study |
| `courses` | 74 | Course catalogue with difficulty and tags |
| `course_offerings` | 74 | Course × Semester instances with curriculum version |
| `enrollments` | 3,920 | Student × Offering registration records |
| `assessments` | 370 | Assessment definitions with weights per offering |
| `grades` | 19,600 | Raw assessment scores per enrollment |
| `attendance` | 54,880 | Weekly attendance records (Present / Absent / Late) |
| `lms_interactions` | 54,880 | Weekly LMS activity — pageviews, video, forum, quizzes |
| `behavior_metrics` | 54,880 | Weekly engagement, attention, inactivity measurements |
| `exam_integrity` | 19,600 | Similarity scores, proctoring flags, sudden-jump flags |
| `outcomes_semester` | 880 | Semester GPA and probation flag per student |
| `student_skills` | 2,008 | Student skill strength ratings (27 skills) |
| `course_skills` | 60 | Skill weights required per course (34 skills) |
| `study_plan_targets` | 220 | Target GPA and risk tolerance per student |
| `course_prereqs` | varies | Course prerequisite relationships |

### 1.2 Entity Relationship Overview

The central join key is `enrollment_id`, which links students to course offerings in a given semester. The bridge join (Section 3.1) maps `enrollment_id → (student_id, semester_id)`, enabling aggregation to the student-semester level required by Goal 2.

> **Fig 1.1 — Entity Relationship Diagram (ERD)**
> ![ERD placeholder](figures/fig_1_1_erd.png)
> *Replace with your ERD export*

---

## 2. Pipeline Architecture

Both goals share a common data loading and feature engineering backbone. They diverge only at the target definition and modelling stage.

> **Fig 2.1 — End-to-end pipeline diagram**
> ![Pipeline placeholder](figures/fig_2_1_pipeline.png)
> *Replace with your flowchart*

### 2.1 Shared Steps

1. Load all 16 tables from SQLite into pandas DataFrames
2. Construct the `enrollment_id → semester_id` bridge join
3. Engineer features at the student-semester level (attendance, LMS, behavior, integrity)
4. Merge student profile features (motivation, stress, target GPA, year of study)
5. Handle missing values and apply log / standard / min-max transformations

### 2.2 Goal-Specific Steps

| | Goal 1 — Regression | Goal 2 — Classification |
|---|---|---|
| **Target** | `weighted_score` (enrollment level) | `next_probation` (student-semester level) |
| **Groupby** | `enrollment_id` | `student_id × semester_id` |
| **Models** | Linear, Ridge, Lasso, ElasticNet, RF, GBM, XGBoost, LightGBM | Logistic Reg, Decision Tree, RF, GBM |
| **Metrics** | RMSE · MAE · R² | AUC · F1 · Precision · Recall |

---

## 3. Feature Engineering

### 3.1 The Bridge Join

Most raw tables are keyed on `enrollment_id` (one row per student-course-semester). Goal 2 requires analysis at the student-semester level. The bridge join resolves this:

```
enrollments  [ enrollment_id, student_id, offering_id ]
     ↕  merge on offering_id
course_offerings  [ offering_id, semester_id, course_id ]
     ↓
bridge  [ enrollment_id, student_id, semester_id ]
```

All enrollment-level tables are then grouped by `(student_id, semester_id)`.

### 3.2 Feature Groups

#### Attendance Features

| Feature | Source | Description |
|---|---|---|
| `att_rate` | `attendance` | Fraction of sessions marked Present |
| `abs_count` | `attendance` | Total absent sessions in the semester |
| `late_rate` | `attendance` | Fraction of sessions marked Late |
| `att_trend` | `attendance` | Δ attendance rate: second half minus first half of semester |

#### LMS Interaction Features

| Feature | Source | Description |
|---|---|---|
| `pageview_mean` | `lms_interactions` | Mean weekly pageviews |
| `minutes_watched_mean` | `lms_interactions` | Mean weekly video minutes watched |
| `forum_posts_sum` | `lms_interactions` | Total forum posts in semester |
| `quiz_attempts_sum` | `lms_interactions` | Total quiz attempts in semester |
| `pv_trend` | `lms_interactions` | Linear slope of weekly pageviews (polyfit deg=1) |
| `mw_trend` | `lms_interactions` | Linear slope of weekly minutes watched |

#### Behavior Features

| Feature | Source | Description |
|---|---|---|
| `engagement_mean` | `behavior_metrics` | Mean weekly engagement score (0–1) |
| `attention_mean` | `behavior_metrics` | Mean weekly attention score (0–1) |
| `inactivity_mean` | `behavior_metrics` | Mean weekly inactivity minutes |

#### Exam Integrity Features

| Feature | Source | Description |
|---|---|---|
| `similarity_mean` | `exam_integrity` | Mean similarity score across assessments |
| `jump_flags` | `exam_integrity` | Total sudden-jump flags in semester |
| `proctor_flags` | `exam_integrity` | Total proctoring flags in semester |

#### Student Profile Features

| Feature | Source | Description |
|---|---|---|
| `motivation_score` | `students` | Self-reported motivation (numeric scale) |
| `stress_level` | `students` | Self-reported stress level (numeric scale) |
| `year_of_study` | `students` | Current year of study |
| `target_gpa` | `study_plan_targets` | Student's self-declared GPA target |
| `risk_tolerance` | `study_plan_targets` | Risk tolerance for course selection |

#### Temporal Features

| Feature | Source | Description |
|---|---|---|
| `current_gpa` | `outcomes_semester` | GPA achieved in the current semester (T) |
| `prev_gpa` | `outcomes_semester` | GPA from the previous semester (T-1) |
| `is_first_semester` | derived | Flag: 1 if no prior semester exists for this student |

> **Note:** `current_gpa` and `prev_gpa` are valid, non-leaky features for Goal 2 because they describe semester T, while the target (`probation_flag`) describes semester T+1.

---

## 4. Goal 1 — Grade Prediction (Regression)

### 4.1 Problem Definition

Given all available features for a student's enrollment, predict their final weighted assessment score for that course offering.

| Property | Value |
|---|---|
| **Target variable** | `weighted_score` — sum of (score × weight) across all assessments |
| **Analysis level** | Enrollment (one row per student-course-semester) |
| **Dataset size** | 3,920 rows · ~140 features after encoding |
| **Target range** | 21.8 – 97.5 (mean ≈ 56.2, std ≈ 11.5) |
| **Train / Test split** | 80% / 20% · random_state=42 |

### 4.2 Target Distribution

> **Fig 4.1 — Histogram of `weighted_score`**
> ![Target distribution placeholder](figures/fig_4_1_target_dist.png)
> *Replace with your plot*

### 4.3 EDA Highlights

> **Fig 4.2 — Feature distributions (numerical)**
> ![Numerical distributions placeholder](figures/fig_4_2_num_dist.png)
> *Replace with your subplot grid*

> **Fig 4.3 — Skewness chart across numerical features**
> ![Skewness placeholder](figures/fig_4_3_skewness.png)
> *Replace with your bar chart*

> **Fig 4.4 — Pearson correlation heatmap (features vs target)**
> ![Correlation heatmap placeholder](figures/fig_4_4_corr_heatmap.png)
> *Replace with your heatmap*

### 4.4 Preprocessing

- **Log transform** applied to: `quiz_attempts_sum`, `forum_posts_sum`, `late_arrival_count`
- **StandardScaler** applied to: trend features, GPA features, `capacity`, `target_gpa`, `minutes_watched_mean`
- **MinMaxScaler** applied to: rate features, count features, `difficulty`
- **OneHotEncoding** (`drop='first'`) applied to all categorical columns
- **Age** derived from `birth_year` — `birth_year` then dropped

### 4.5 Models Evaluated

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

### 4.6 Evaluation Metrics — Goal 1

All models evaluated on the held-out 20% test set.

| Model | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|
| Linear Regression | — | — | — |
| Ridge Regression | — | — | — |
| Lasso Regression | — | — | — |
| ElasticNet Regression | — | — | — |
| Decision Tree | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |
| XGBoost | — | — | — |
| LightGBM | — | — | — |

> ↓ lower is better · ↑ higher is better

### 4.7 Key Observations — Goal 1

> *Fill this section after running your notebook. Suggested talking points:*
> - Which model achieved the lowest RMSE / highest R²?
> - Did ensemble methods (RF, GBM) substantially outperform linear baselines?
> - Was R² low overall? Note that synthetic data may have weak feature-target signal by design.
> - Which features correlated most strongly with `weighted_score`?

> **Fig 4.5 — Predicted vs Actual scatter plot (best model)**
> ![Predicted vs actual placeholder](figures/fig_4_5_pred_vs_actual.png)
> *Replace with your plot*

---

## 5. Goal 2 — Failure Risk Classification

### 5.1 Problem Definition

Given a student's full academic profile and behavior signals from semester T, predict whether they will be placed on academic probation in semester T+1.

| Property | Value |
|---|---|
| **Target variable** | `next_probation` — `probation_flag` shifted forward one semester |
| **Analysis level** | Student × Semester (bridge join required) |
| **Dataset size** | 660 rows after requiring a following semester to exist |
| **Class balance** | ~68% on probation · ~32% not on probation |
| **Train / Test split** | 80% / 20% · stratified by target · random_state=42 |
| **CV strategy** | 5-fold StratifiedKFold |

> **Leakage note:** `current_gpa` (semester T outcome) is a valid feature — it describes a completed past semester. Using `avg_score` from the **same** semester as the target would constitute leakage and is excluded.

### 5.2 Target Distribution

> **Fig 5.1 — Class balance bar chart and pie chart**
> ![Class balance placeholder](figures/fig_5_1_class_balance.png)
> *Replace with your plot*

### 5.3 EDA Highlights

> **Fig 5.2 — Feature distributions by probation status**
> ![Feature distributions placeholder](figures/fig_5_2_feat_dist.png)
> *Replace with your overlapping histogram subplot grid*

> **Fig 5.3 — Correlation heatmap — top 15 features vs `probation_flag`**
> ![Correlation heatmap placeholder](figures/fig_5_3_corr_heatmap.png)
> *Replace with your heatmap*

### 5.4 Models Evaluated

- Logistic Regression (C=0.5, max_iter=1000)
- Decision Tree Classifier (max_depth=6)
- Random Forest Classifier (n_estimators=200)
- Gradient Boosting Classifier (n_estimators=200, learning_rate=0.05)

### 5.5 Evaluation Metrics — Goal 2

#### Cross-Validation Results (5-Fold Stratified)

| Model | AUC ↑ | F1 ↑ | Precision ↑ | Recall ↑ |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Decision Tree | — | — | — | — |
| Random Forest | — | — | — | — |
| Gradient Boosting | — | — | — | — |

#### Best Model — Test Set Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| No Probation | — | — | — | — |
| Probation | — | — | — | — |
| Macro avg | — | — | — | — |
| Weighted avg | — | — | — | — |

**ROC-AUC (test set):** —  &nbsp;&nbsp; **Accuracy:** —

### 5.6 Model Diagnostics

> **Fig 5.4 — Model comparison bar chart (AUC / F1 / Precision / Recall)**
> ![Model comparison placeholder](figures/fig_5_4_model_comparison.png)
> *Replace with your plot*

> **Fig 5.5 — Confusion matrix**
> ![Confusion matrix placeholder](figures/fig_5_5_confusion_matrix.png)
> *Replace with your heatmap*

> **Fig 5.6 — ROC Curve**
> ![ROC curve placeholder](figures/fig_5_6_roc_curve.png)
> *Replace with your plot*

> **Fig 5.7 — Precision-Recall Curve**
> ![PR curve placeholder](figures/fig_5_7_pr_curve.png)
> *Replace with your plot*

### 5.7 Feature Importance & Explainability

> **Fig 5.8 — Feature importance bar chart (Gradient Boosting, Gini)**
> ![Feature importance placeholder](figures/fig_5_8_feature_importance.png)
> *Replace with your plot*

> **Fig 5.9 — Risk profile: mean feature value by class**
> ![Risk profile placeholder](figures/fig_5_9_risk_profile.png)
> *Replace with your grouped bar chart*

#### Top Influencing Factors

> *Fill after running your notebook. Suggested structure:*
> - **Feature 1** (e.g. `current_gpa`) — brief explanation of direction and magnitude
> - **Feature 2** (e.g. `pv_trend`) — brief explanation
> - **Feature 3** (e.g. `stress_level`) — brief explanation
> - **Feature 4** (e.g. `att_trend`) — brief explanation
> - **Feature 5** (e.g. `inactivity_mean`) — brief explanation

### 5.8 Key Observations — Goal 2

> *Fill after running your notebook. Suggested talking points:*
> - Which model achieved the best AUC? Was there a precision / recall tradeoff worth discussing?
> - How much does `current_gpa` alone explain versus behavioral signals?
> - Did LMS trend features (`pv_trend`, `mw_trend`) add meaningful signal beyond raw averages?
> - **Note on synthetic data:** behavioral signals show weak predictive power because the dataset generator did not encode a causal relationship between engagement and GPA. This is expected and worth stating explicitly.

---

## 6. Limitations & Future Work

### 6.1 Known Limitations

- **Synthetic data:** behavioral features (attendance, LMS, engagement) were generated independently from GPA. This artificially suppresses the predictive signal that would exist in real-world data.
- **Small dataset:** 660 student-semester rows for Goal 2 limits model complexity and generalisability.
- **Class imbalance:** ~68% positive rate in Goal 2 inflates F1 for the majority class — Precision for the minority (No Probation) class is lower.
- **No temporal validation:** a production model should be validated on a future held-out semester, not a random split.
- **Goal 1 R² ceiling:** regression performance is bounded by the weak feature-target correlation baked into the synthetic data.

### 6.2 Future Work

- **Temporal cross-validation:** train on semesters 1–3, validate on semester 4.
- **SMOTE or class weighting** to address imbalance in Goal 2.
- **SHAP values** for deeper per-student explainability beyond aggregate feature importance.
- **LSTM / sequence model** for Goal 1 using weekly grade trajectories as a time series.
- **Hybrid recommendation system** (Goal 5) using `student_skills × course_skills` cosine similarity.

---

## 7. Repository Structure

```
edu_exam.sqlite                  ← source database
student_grades_model.ipynb       ← Goal 1: regression notebook
student_risk_model.ipynb         ← Goal 2: classification notebook
README.md                        ← this file
figures/
  ├── fig_1_1_erd.png
  ├── fig_2_1_pipeline.png
  ├── fig_4_1_target_dist.png
  ├── fig_4_2_num_dist.png
  ├── fig_4_3_skewness.png
  ├── fig_4_4_corr_heatmap.png
  ├── fig_4_5_pred_vs_actual.png
  ├── fig_5_1_class_balance.png
  ├── fig_5_2_feat_dist.png
  ├── fig_5_3_corr_heatmap.png
  ├── fig_5_4_model_comparison.png
  ├── fig_5_5_confusion_matrix.png
  ├── fig_5_6_roc_curve.png
  ├── fig_5_7_pr_curve.png
  ├── fig_5_8_feature_importance.png
  └── fig_5_9_risk_profile.png
```

---

## 8. Dependencies

| Package | Version (tested) | Usage |
|---|---|---|
| `pandas` | ≥ 2.0 | Data loading, merging, aggregation |
| `numpy` | ≥ 1.24 | Numerical operations, polyfit trends |
| `scikit-learn` | ≥ 1.3 | Preprocessing, models, metrics, CV |
| `xgboost` | ≥ 2.0 | XGBoost regressor (Goal 1) |
| `lightgbm` | ≥ 4.0 | LightGBM regressor (Goal 1) |
| `matplotlib` | ≥ 3.7 | All plots |
| `seaborn` | ≥ 0.13 | Heatmaps |
| `scipy` | ≥ 1.11 | Skewness calculation |
| `sqlite3` | stdlib | Database connection |

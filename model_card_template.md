# Model Card

## Model Details
This is a supervised machine learning model for **binary classification** that predicts whether a person’s annual income is **>50K** or **<=50K** based on demographic and employment-related features from the Census Income (Adult) dataset.

- **Model type:** RandomForestClassifier (scikit-learn)
- **Task:** Binary classification
- **Target label:** `salary` mapped to {">50K", "<=50K"}
- **Pipeline preprocessing:**
  - One-hot encoding for categorical features using `OneHotEncoder(handle_unknown="ignore")`
  - Label binarization using `LabelBinarizer()`
- **Training configuration:**
  - `n_estimators=200`
  - `random_state=42`
  - `n_jobs=-1`

## Intended Use
**Primary intended use:** Educational purposes and demonstration of a scalable ML pipeline deployed with FastAPI, including training, inference, and subgroup (slice) performance evaluation.

**Intended users:** Students, reviewers, and practitioners learning MLOps fundamentals.

**Out-of-scope use:** This model must **not** be used for real-world decision-making in high-stakes domains (e.g., hiring, credit approval, insurance, benefits), since it is trained on historical census data that may encode societal biases and has not been validated for deployment in production decision systems.

## Factors
The model may exhibit different performance across subgroups defined by categorical factors present in the dataset, including (but not limited to):

- `sex`
- `race`
- `native-country`
- `education`
- `workclass`
- `occupation`
- `marital-status`
- `relationship`

Slice-based performance evaluation is reported per value of each categorical feature.

## Metrics
The model is evaluated using:
- **Precision**
- **Recall**
- **F1-score** (implemented as F-beta with `beta=1`)

### Overall Test Performance (20% held-out test set)
The following results were obtained on the stratified 20% test split:

- **Precision:** 0.7338  
- **Recall:** 0.6365  
- **F1-score:** 0.6817  

### Slice-Based Performance
Performance varies significantly across subgroups.

- **Best-performing large subgroup:**  
  occupation = **Prof-specialty** (n = 818)  
  Precision: 0.7793 | Recall: 0.8139 | F1-score: **0.7962**

- **Worst-performing subgroup with meaningful size:**  
  education = **7th–8th** (n = 120)  
  Precision: 1.0000 | Recall: 0.0000 | F1-score: **0.0000**

Across slices, F1-score ranged approximately from **0.00 to ~0.80**, indicating substantial performance disparities across demographic and socioeconomic groups. Several very small-population country groups showed extreme values (F1 = 0 or 1), but these results may be unstable due to very low sample sizes.

## Evaluation Data
Evaluation is performed on a **20% stratified hold-out test split** from the Census Income (Adult) dataset. The test set is not used during model training.

In addition to overall evaluation, the model is evaluated on **categorical slices** to detect subgroup performance differences. Slice results are stored in `slice_output.txt`.

## Training Data
Training uses the **Census Income (Adult) dataset**, where each record contains demographic and employment-related attributes such as:

age, workclass, education, marital-status, occupation, relationship, race, sex, hours-per-week, native-country, etc.

The training set corresponds to **80%** of the dataset, stratified by the target label.

## Ethical Considerations
This dataset contains sensitive attributes (e.g., sex, race, native-country). As a result:
- The model may learn and reproduce historical bias patterns.
- Performance disparities across subgroups are observed.
- The model could be misused in ways that unfairly impact individuals.

Therefore, this model should not be used for real-world decisions that directly impact individuals without extensive fairness analysis, bias mitigation, and careful validation.

## Caveats and Recommendations
- The model was trained on a specific dataset and may not generalize well to other populations or time periods.
- Slice-based evaluation shows substantial disparities (e.g., very low performance for some education and country subgroups), which should be reviewed before any deployment.
- Future improvements could include:
  - hyperparameter tuning,
  - alternative models and calibration,
  - bias mitigation techniques,
  - improved handling of low-sample subgroups,
  - more detailed documentation of data provenance and intended deployment constraints.

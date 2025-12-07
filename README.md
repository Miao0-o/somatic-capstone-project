# somatic-capstone-project

**Author:** Miao Yu 
**Project Type:** Data Science / Machine Learning  
**Tech Stack:** Python, TensorFlow/Keras, Pandas, Scikit-learn, Matplotlib 

## Project Overview
Fatigue is one of the most common somatic symptoms reported in both clinical and community populations. It affects daily functioning and quality of life, and is influenced by multiple psychosocial and demographic factors.

This project aims to **predict the presence of fatigue** using a feedforward neural network model trained on the publicly available **EAMMi2 dataset** (Grahe et al., 2018).  

The goal is to demonstrate practical machine learning skills, from data preprocessing to model evaluation and interpretation.

---

## Dataset
- **Source:** [EAMMi2 Dataset – Open Psychology Data](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.38/)
- **Size:** ~3000+ participants  
- **Features:** Psychosocial scale scores, demographic information  
- **Target Variable:** Binary indicator of fatigue (PHQ symptom item 12, recoded)

**Note:**  
The dataset is included in `data/` for reproducibility. Only variables necessary for model training are kept.

---

## Methods
1. **Data Cleaning & Preprocessing**
   - Handling missing values with `SimpleImputer`
   - Scaling features with `StandardScaler`
   - Stratified train-test split (70% / 30%)

2. **Model**
   - Framework: TensorFlow’s Keras API
   - Architecture:
     - Input layer: 24 features
     - Hidden layers: 64 → 32 units, ReLU activation
     - Dropout (rate=0.3) to reduce overfitting
     - Output layer: Sigmoid activation (binary classification)
   - Optimizer: Adam (learning rate=0.001)
   - Loss: Binary cross-entropy

3. **Evaluation**
   - Accuracy
   - AUC (Area Under the ROC Curve)
   - Confusion matrix
   - ROC curve visualization
   - Training history plots
   
1. **Data Cleaning, Outcome Coding, and Sample Definition**
  - Source data: EAMMi2 public dataset (PHQ-15 items, psychosocial scales, and demographics).
  - Each PHQ-15 item was recoded to a binary outcome:
  - 0 = “not bothered at all”
  - 1 = “bothered a little” or “bothered a lot”.
  - A separate analytic dataset was created for each of the 13 symptoms; cases with missing data on the outcome or selected predictors were removed for that symptom (complete-case per outcome).

2. **Psychological Scales and Feature Engineering**
 - Multi-item psychological constructs (e.g., perceived stress, social support, need to belong, mindfulness, subjective well-being, self-efficacy, identity exploration, disability identity, social media scales, interpersonal exploitativeness/transgressions, narcissistic traits, American Dream beliefs) were scored as mean composites.
 - Additional summary features (e.g., within-scale standard deviation, minimum, maximum) were created to capture variability within each construct.
 - A set of theory-guided engineered predictors was derived to reflect risk/protection patterns (e.g., combinations of high stress and low support/mindfulness, indices of distress vs. protective factors), while keeping features interpretable and aligned with psychological theory.
 - The final predictor set combined these psychological features with demographic variables (e.g., sex, race/ethnicity, income, education, parental marital status, sibling structure).

3. **Preprocessing Pipeline**
 - Categorical predictors were dummy-coded / one-hot encoded using scikit-learn transformers.
 - Continuous predictors (scale scores and engineered features) were standardized (mean = 0, SD = 1) within the training data to support model stability and comparability across algorithms.
 - For each symptom, data were split into stratified train–test sets (typically 80/20), preserving the proportion of symptomatic vs. non-symptomatic cases.
 - All preprocessing steps (imputation if used, encoding, scaling) were implemented inside scikit-learn Pipelines to avoid data leakage.

4. **Models Compared**
 - For each symptom, we fit and compared a set of classical machine-learning models and a neural-network–style model, including:
 - Penalized logistic regression (L1 / elastic net)
 - K-nearest neighbors (KNN)
 - Tree-based models:
    - Random forest
    - Gradient boosting models
    - ExtraTrees
 - TabNet‐style neural network for tabular data
 - In some analyses, an ensemble / stacking model was used to combine strong base learners.
**The best model for each symptom was selected based on validation performance.**

5. **Evaluation Metrics**
 - Because symptom prevalence was often imbalanced (some symptoms common, others rare), we used metrics that go beyond simple accuracy:
 - ROC–AUC to assess overall discriminability across thresholds.
 - F1 score to balance precision and recall for the “symptom present” class.
 - Balanced accuracy in earlier stages to account for class imbalance.
 - In the final interpretation, particular emphasis was placed on ROC–AUC and F1. Only a subset of symptoms (especially fatigue and trouble sleeping) reached high scores on both metrics, indicating reliable predictability from psychosocial and demographic features.

6. **Model Interpretability (SHAP)**
 - For each symptom, we computed SHAP values for the best-performing model to estimate feature importance and visualize how specific predictors affected symptom risk.
 - Due to weaker performance for many symptoms, detailed SHAP interpretation on the poster and paper focused on fatigue and sleep problems, where models showed the most robust F1 and ROC–AUC.
 - SHAP plots highlight contributions of variables such as stress, mindfulness, subjective well-being, need to belong, interpersonal exploitativeness, and demographic factors to individual symptom risk.

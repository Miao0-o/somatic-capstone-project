# somatic-capstone-project

**Author:** Miao Yu, Jessica Tanchone 

**Project Type:** Data Science / Machine Learning  

**Tech Stack:** Python, TensorFlow/Keras, Pandas, Scikit-learn, Matplotlib 

## Project Overview
This project aims to predict **fatigue**, a common somatic symptom, using psychosocial and demographic variables data.  
Two main modeling approaches are implemented and compared:
1. **Traditional machine learning models** (Random Forest, Logistic Regression, K-Nearest Neighbors)
2. **Neural network models** (Feedforward network with Keras/TensorFlow)

The goal is to evaluate which modeling approach yields better predictive performance and identify the most important predictors of fatigue.


---

## Dataset
- **Source:** Since this is an in-progress project, the data source will not be made public until the project is completed.
- **Size:** ~over 3000 participants  
- **Features:** Psychosocial scale scores, demographic information  
- **Target Variable:**  Binary indicators for 13 symptoms from the PHQ questionnaire

**Note:**  
The dataset is included in `data/` for reproducibility. Only variables necessary for model training are kept.

---

## Methods

### 1. **Data Preprocessing**
- Missing values handled with mean/mode imputation
- Categorical variables one-hot encoded
- Continuous variables scaled using `StandardScaler`
- Addressed class imbalance with **SMOTE** oversampling

### 2. **Models Implemented**
- **Logistic Regression** (baseline)
- **Random Forest Classifier**
- **K-Nearest Neighbors**
- **Feedforward Neural Network** (2 hidden layers, dropout regularization)

### 3. **Evaluation Metrics**
- Accuracy
- Balanced Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix

3. **Evaluation**
   - Accuracy
   - AUC (Area Under the ROC Curve)
   - Confusion matrix
   - ROC curve visualization
   - Training history plots

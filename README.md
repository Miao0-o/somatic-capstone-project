# somatic-capstone-project

**Author:** Miao Yu 
**Project Type:** Data Science / Machine Learning  
**Tech Stack:** Python, TensorFlow/Keras, Pandas, Scikit-learn, Matplotlib 

## 📌 Project Overview
Fatigue is one of the most common somatic symptoms reported in both clinical and community populations. It affects daily functioning and quality of life, and is influenced by multiple psychosocial and demographic factors.

This project aims to **predict the presence of fatigue** using a feedforward neural network model trained on the publicly available **EAMMi2 dataset** (Grahe et al., 2018).  

The goal is to demonstrate practical machine learning skills, from data preprocessing to model evaluation and interpretation.

---

## 📊 Dataset
- **Source:** [EAMMi2 Dataset – Open Psychology Data](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.38/)
- **Size:** ~2,500 participants  
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

# Stroke-Risk-Prediction

## Project Overview
This project aims to predict stroke risk using machine learning techniques based on a dataset containing demographic, medical history, and lifestyle factors Given the highly imbalanced nature of stroke occurrences in the dataset, special attention was given to data preprocessing, feature selection, and balancing techniques** to improve predictive accuracy. The ultimate goal is to develop a reliable and interpretable model that can assist healthcare professionals and individuals in identifying high-risk patients and promoting early preventive measures.

## Key Features
- **Exploratory Data Analysis (EDA)** to understand stroke distributions and key contributing factors.
- **Handling missing values** and applying **feature transformations** to normalize skewed data.
- **Class imbalance mitigation** using **SMOTE, undersampling, and class weighting techniques**.
- **Comparison of multiple classification models**, including **Logistic Regression, Support Vector Machines (SVM), and Stochastic Gradient Descent (SGD)**.
- **Model evaluation using MCC (Matthews Correlation Coefficient), AUC (Area Under Curve), and Recall** to prioritize stroke case identification.
- **Hyperparameter tuning** to optimize model performance.
- **Ensemble techniques and voting classifiers** to improve predictive stability.
- **Visualizations** showcasing data distributions, transformations, and model performance metrics.

## Dataset
The dataset used in this project is sourced from **[Kaggle’s Stroke Prediction Dataset](https://www.kaggle.com/datasets/lirilkumaramal/heart-stroke)**. It contains patient records with **features relevant to stroke risk assessment**, including **demographics, health history, and lifestyle habits**.

### Key Features
- **Demographic Attributes**: Age, Gender, Residence Type (Urban/Rural).
- **Medical History**: Hypertension, Heart Disease, and Previous Stroke Occurrence.
- **Lifestyle Factors**: Smoking Status, Work Type.
- **Health Indicators**: BMI, Average Glucose Level.
- **Target Variable**: Stroke (Binary 0/1 indicating stroke occurrence).

### Key Columns
```plaintext
['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
```

## Methods and Tools
### Data Preparation
- **Handling Missing Values**: Imputed missing **BMI** values using the **median** and created an **"Unknown" category** for missing smoking status.
- **Feature Engineering**: One-hot encoding for categorical variables such as **work_type** and **smoking_status**.
- **Data Transformation**: Applied **Box-Cox transformation** to **glucose levels, BMI, and age** to normalize skewed distributions.
- **Class Balancing Techniques**:
  - **SMOTE (Synthetic Minority Over-sampling Technique)**
  - **Undersampling the majority class**
  - **Using class weighting in models** instead of explicit resampling.

### Modeling
- **Model Selection**: Multiple models were tested, including:
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - Linear SVC
  - Stochastic Gradient Descent (SGD)
- **Evaluation Metrics**:
  - **Matthews Correlation Coefficient (MCC)** – Ideal for imbalanced binary classification.
  - **AUC (Area Under Curve)** – Measures trade-off between True Positive and False Positive rates.
  - **Recall** – Prioritizes correct stroke predictions, minimizing false negatives.
- **Hyperparameter Tuning**:
  - **GridSearchCV** for tuning SVC and SGD.
  - **RandomizedSearchCV** for optimizing the Multi-layer Perceptron (MLP) model.
- **Ensemble Learning**:
  - **VotingClassifier** combining multiple models for improved stability.

## Results
- **Stroke cases were highly correlated with age, hypertension, heart disease, and glucose levels**.
- **Self-employed individuals** showed a higher stroke risk than other work types.
- **Former smokers had a stronger association with strokes than current smokers**.
- **Balancing the dataset through class weighting yielded better results than oversampling or undersampling**.
- **SVM-based models and Logistic Regression performed best**, with MCC scores > 0.15 and AUC > 0.85.

## Installation
To replicate the project:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/football-performance-prediction.git
   ```



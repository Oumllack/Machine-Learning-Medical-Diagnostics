# Deep Learning and Clustering Analysis for Diabetes Risk Prediction
## A Comprehensive Study on the Pima Indians Diabetes Dataset

![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Abstract

This study presents a comprehensive analysis of diabetes prediction and risk stratification using advanced machine learning techniques on the Pima Indians Diabetes dataset. We implemented multiple classification models, including deep neural networks, and performed unsupervised clustering to identify distinct patient subgroups. Our best model achieved 81.5% accuracy, while clustering revealed three distinct patient profiles with varying diabetes risk levels. This research provides valuable insights for early diabetes detection and personalized healthcare strategies.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Methodology](#methodology)
4. [Results and Analysis](#results-and-analysis)
5. [Discussion](#discussion)
6. [Conclusions and Future Work](#conclusions-and-future-work)
7. [Technical Implementation](#technical-implementation)

## Introduction

Diabetes mellitus is a global health concern affecting millions worldwide. Early detection and risk stratification are crucial for effective prevention and management. This study leverages machine learning and deep learning techniques to:
- Predict diabetes status with high accuracy
- Identify distinct patient subgroups
- Provide actionable insights for healthcare professionals

## Dataset Description

### Overview
The Pima Indians Diabetes dataset contains medical records of 768 female patients of Pima Indian heritage, including:
- 8 physiological features
- Binary outcome (diabetes diagnosis)
- No missing values after preprocessing

### Features
1. Pregnancies: Number of times pregnant
2. Glucose: Plasma glucose concentration (mg/dL)
3. BloodPressure: Diastolic blood pressure (mm Hg)
4. SkinThickness: Triceps skin fold thickness (mm)
5. Insulin: 2-Hour serum insulin (mu U/ml)
6. BMI: Body mass index (kg/m²)
7. DiabetesPedigreeFunction: Diabetes pedigree function
8. Age: Age in years
9. Outcome: Diabetes diagnosis (1 = positive, 0 = negative)

### Data Distribution
- Class distribution: 65% non-diabetic, 35% diabetic
- Age range: 21-81 years
- BMI range: 18.2-67.1 kg/m²

## Methodology

### 1. Data Preprocessing
- Zero value replacement with median
- Feature scaling (MinMaxScaler)
- Class balancing using SMOTE
- Train-test split (80-20)

### 2. Classification Models
We implemented and compared multiple models:

#### 2.1 Deep Neural Networks
- MLP (4 layers): 256-128-64-32 neurons
- MLP (5 layers): 512-256-128-64-32 neurons
- Activation: ReLU
- Optimizer: Adam
- Early stopping with patience=10
- Learning rate adaptation

#### 2.2 Traditional Machine Learning
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Stacking Classifier

### 3. Clustering Analysis
- K-means clustering
- Optimal clusters determined by silhouette score
- Feature standardization
- Cluster profiling

## Results and Analysis

### 1. Classification Performance

#### Model Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|-----------|
| MLP (5 layers) | 81.5% | 0.82 | 0.81 | 0.81 |
| MLP (4 layers) | 81.0% | 0.82 | 0.81 | 0.81 |
| LightGBM | 78.0% | 0.79 | 0.78 | 0.78 |
| XGBoost | 74.0% | 0.75 | 0.74 | 0.74 |
| Random Forest | 73.0% | 0.74 | 0.73 | 0.73 |
| Logistic Regression | 70.0% | 0.71 | 0.70 | 0.70 |

### 2. Clustering Results

#### Cluster Profiles
| Cluster | Size | Age | Glucose | BMI | Insulin | Diabetes Rate |
|---------|------|-----|---------|-----|---------|---------------|
| 0 | 256 | 29.3 | 141.4 | 39.2 | 206.0 | 55% |
| 1 | 256 | 45.6 | 129.7 | 32.9 | 136.0 | 51% |
| 2 | 256 | 26.0 | 106.0 | 28.8 | 111.8 | 13% |

#### Key Findings
1. **High-Risk Cluster (0)**
   - Young patients (29.3 years)
   - High glucose levels (141.4 mg/dL)
   - High BMI (39.2 kg/m²)
   - Highest diabetes rate (55%)

2. **Moderate-Risk Cluster (1)**
   - Middle-aged patients (45.6 years)
   - Moderate glucose levels (129.7 mg/dL)
   - Normal BMI (32.9 kg/m²)
   - Moderate diabetes rate (51%)

3. **Low-Risk Cluster (2)**
   - Young patients (26.0 years)
   - Normal glucose levels (106.0 mg/dL)
   - Normal BMI (28.8 kg/m²)
   - Lowest diabetes rate (13%)

## Discussion

### Model Performance
- The deep neural network (MLP) achieved the best performance (81.5% accuracy)
- Balanced precision and recall across classes
- Robust performance on test set
- No significant overfitting observed

### Risk Stratification
1. **High-Risk Group**
   - Young patients with high glucose and BMI
   - Immediate intervention recommended
   - Regular monitoring required

2. **Moderate-Risk Group**
   - Middle-aged patients with moderate glucose levels
   - Lifestyle modifications recommended
   - Regular check-ups advised

3. **Low-Risk Group**
   - Young patients with normal metrics
   - Standard preventive care
   - Annual screening recommended

### Clinical Implications
1. **Early Detection**
   - Model can identify high-risk patients before diagnosis
   - Enables proactive healthcare interventions

2. **Personalized Care**
   - Cluster-based risk stratification
   - Tailored prevention strategies

3. **Resource Allocation**
   - Efficient targeting of healthcare resources
   - Priority-based patient management

## Conclusions and Future Work

### Key Conclusions
1. Deep learning models outperform traditional ML in diabetes prediction
2. Three distinct patient subgroups identified
3. High accuracy achieved without overfitting
4. Actionable insights for healthcare providers

### Future Work
1. **Model Improvements**
   - Ensemble of deep learning models
   - Advanced feature engineering
   - Integration of additional medical data

2. **Clinical Validation**
   - Prospective studies
   - Real-world implementation
   - Multi-center validation

3. **Technical Enhancements**
   - Real-time prediction system
   - Mobile application development
   - Integration with electronic health records

## Technical Implementation

### Requirements
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.6.1
tensorflow==2.15.0
optuna==3.5.0
imbalanced-learn==0.11.0
xgboost==2.0.0
lightgbm==4.1.0
matplotlib==3.7.2
seaborn==0.12.2
```

### Project Structure
```
.
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── deep_learning_experiments.ipynb
├── deep_train.py
├── requirements.txt
└── README.md
```

### Usage
1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run analysis:
```bash
python deep_train.py
```

### Output Files
- `cluster_analysis_deep.csv`: Detailed cluster analysis
- `data_with_clusters.csv`: Original data with cluster assignments
- `patients_diabetiques.csv`: Identified diabetic patients
- `patients_a_risque.csv`: High-risk patients

## Acknowledgments
- Pima Indians Diabetes Dataset (UCI Machine Learning Repository)
- Scikit-learn, TensorFlow, and other open-source libraries
- Research community for valuable insights and methodologies

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or collaborations, please open an issue in this repository.

---

*Note: This study is for research purposes only. Medical decisions should be made by qualified healthcare professionals.* 
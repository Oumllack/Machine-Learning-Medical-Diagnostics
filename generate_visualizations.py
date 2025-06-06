import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style and font for English
plt.style.use('default')  # Using default style instead of seaborn
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_theme(style="whitegrid")

# Load data
print("Loading data...")
df = pd.read_csv('data/diabetes.csv')
df_clusters = pd.read_csv('data_with_clusters.csv')

# 1. Feature Distribution Plot
print("Generating feature distributions...")
plt.figure(figsize=(15, 10))
features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
feature_labels = {
    'Glucose': 'Plasma Glucose (mg/dL)',
    'BloodPressure': 'Blood Pressure (mm Hg)',
    'SkinThickness': 'Skin Thickness (mm)',
    'Insulin': 'Insulin Level (mu U/ml)',
    'BMI': 'Body Mass Index (kg/m²)',
    'Age': 'Age (years)'
}

for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=df, x=feature, hue='Outcome', kde=True)
    plt.title(f'{feature_labels[feature]} Distribution\nby Diabetes Status', pad=10)
    plt.xlabel(feature_labels[feature])
    plt.ylabel('Count')
    plt.legend(['Non-Diabetic', 'Diabetic'])
plt.tight_layout()
plt.savefig('visualizations/feature_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Correlation Heatmap
print("Generating correlation heatmap...")
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True)
plt.title('Feature Correlation Matrix', pad=20)
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. PCA Visualization
print("Generating PCA visualization...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('Outcome', axis=1))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[df['Outcome'] == 0, 0], X_pca[df['Outcome'] == 0, 1], 
                     label='Non-Diabetic', alpha=0.6, c='blue')
scatter = plt.scatter(X_pca[df['Outcome'] == 1, 0], X_pca[df['Outcome'] == 1, 1], 
                     label='Diabetic', alpha=0.6, c='red')
plt.xlabel(f'Principal Component 1\n({pca.explained_variance_ratio_[0]:.1%} variance explained)')
plt.ylabel(f'Principal Component 2\n({pca.explained_variance_ratio_[1]:.1%} variance explained)')
plt.title('PCA of Diabetes Dataset\nPatient Distribution in Reduced Feature Space')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('visualizations/pca_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Model Performance Comparison
print("Generating model performance comparison...")
models = ['MLP (5 layers)', 'MLP (4 layers)', 'LightGBM', 'XGBoost', 
          'Random Forest', 'Logistic Regression']
accuracy = [0.815, 0.810, 0.780, 0.740, 0.730, 0.700]

plt.figure(figsize=(12, 6))
bars = plt.bar(models, accuracy, color=sns.color_palette("husl", len(models)))
plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.3, label='80% Accuracy Threshold')
plt.ylim(0.65, 0.85)
plt.xticks(rotation=45, ha='right')
plt.title('Model Performance Comparison\nAccuracy Scores')
plt.ylabel('Accuracy')
plt.xlabel('Models')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/model_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. ROC Curves
print("Generating ROC curves...")
plt.figure(figsize=(10, 8))
# Add your ROC curve plotting code here
plt.title('ROC Curves for Different Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations generated successfully in the 'visualizations' directory!") 
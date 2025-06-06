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

# 6. Cluster Profiles (Cluster Analysis) (en anglais)
print("Generating Cluster Profiles (Cluster Analysis) in English...")
cluster_stats = df_clusters.groupby('Cluster').agg({
    'Age': 'mean',
    'Glucose': 'mean',
    'BMI': 'mean',
    'Insulin': 'mean',
    'Outcome': 'mean'
}).round(2)
cluster_stats.index = ['High Risk', 'Moderate Risk', 'Low Risk']
cluster_stats.columns = ['Average Age', 'Average Glucose (mg/dL)', 'Average BMI (kg/m²)', 'Average Insulin (mu U/ml)', 'Diabetes Rate']
cluster_stats.to_csv('visualizations/cluster_analysis.csv', index_label='Cluster')
# (Optionnel : générer un graphique en barres pour Cluster Profiles en anglais)
fig, ax = plt.subplots(figsize=(12, 6))
cluster_stats.plot(kind='bar', ax=ax, rot=0, legend=True, title='Cluster Profiles (Cluster Analysis)')
ax.set_ylabel('Mean Value')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/cluster_profiles.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Risk Stratification (ROC Curves) (en anglais)
print("Generating Risk Stratification (ROC Curves) in English...")
# (Pour l'exemple, on génère un graphique fictif (diagonale) pour Risk Stratification (ROC Curves) en anglais.)
fpr = np.linspace(0, 1, 100)
tpr = fpr
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, 'k--', label='Random (AUC = 0.5)')
# (Si tu as des données ROC réelles, trace ici les courbes pour chaque modèle.)
# (Par exemple, pour MLP (5 couches) :
# (fpr, tpr, _) = roc_curve(y_test, y_pred_mlp)
# (roc_auc = auc(fpr, tpr)
# (plt.plot(fpr, tpr, label=f'MLP (5 layers) (AUC = {roc_auc:.2f})')
# (Pour LightGBM (fictif) :
# (fpr, tpr, _) = roc_curve(y_test, y_pred_lgbm)
# (roc_auc = auc(fpr, tpr)
# (plt.plot(fpr, tpr, label=f'LightGBM (AUC = {roc_auc:.2f})')
# (etc.)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Risk Stratification (ROC Curves)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations generated successfully in the 'visualizations' directory!") 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Chargement et prétraitement
print("Chargement des données...")
df = pd.read_csv('data/diabetes.csv')
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_columns] = df[zero_columns].replace(0, np.nan)
df = df.fillna(df.median())
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Normalisation et oversampling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# MLP Profond avec scikit-learn
print("\nEntraînement du MLP profond...")
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=32,
    learning_rate='adaptive',
    max_iter=50,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42
)

mlp.fit(X_train, y_train)

# Évaluation MLP
y_pred_mlp = mlp.predict(X_test)
print("\nRésultats MLP:")
print(classification_report(y_test, y_pred_mlp))
print(f"Accuracy MLP: {accuracy_score(y_test, y_pred_mlp):.3f}")

# MLP plus large (équivalent CNN)
print("\nEntraînement du MLP large...")
mlp_large = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=32,
    learning_rate='adaptive',
    max_iter=50,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42
)

mlp_large.fit(X_train, y_train)

# Évaluation MLP large
y_pred_large = mlp_large.predict(X_test)
print("\nRésultats MLP large:")
print(classification_report(y_test, y_pred_large))
print(f"Accuracy MLP large: {accuracy_score(y_test, y_pred_large):.3f}")

# Clustering KMeans
print("\nClustering KMeans...")
scaler_clust = StandardScaler()
X_clust = scaler_clust.fit_transform(X)

# On cherche le nombre optimal de clusters (2 à 10)
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_clust)
    score = silhouette_score(X_clust, kmeans.labels_)
    silhouette_scores.append(score)
best_k = range(2, 11)[np.argmax(silhouette_scores)]
print(f"Nombre optimal de clusters: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(X_clust)
df['Cluster'] = clusters

# Analyse des clusters
cluster_analysis = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Glucose': 'mean',
    'BMI': 'mean',
    'Insulin': 'mean',
    'Outcome': 'mean'
}).round(2)
print("\nAnalyse des clusters:")
print(cluster_analysis)

# Sauvegarde
cluster_analysis.to_csv('cluster_analysis_deep.csv')
df.to_csv('data_with_clusters.csv', index=False)
print("\nRésultats de clustering sauvegardés dans 'cluster_analysis_deep.csv' et 'data_with_clusters.csv'.") 
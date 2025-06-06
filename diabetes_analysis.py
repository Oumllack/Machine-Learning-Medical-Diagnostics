#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
RANDOM_STATE = 42
DATA_PATH = 'data/diabetes.csv'

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

def create_advanced_features(df):
    """Crée des features avancées pour améliorer les performances."""
    # Features existantes
    df_new = df.copy()
    
    # Interactions entre variables importantes
    df_new['Glucose_BMI'] = df_new['Glucose'] * df_new['BMI']
    df_new['Age_BMI'] = df_new['Age'] * df_new['BMI']
    df_new['Glucose_Age'] = df_new['Glucose'] * df_new['Age']
    
    # Ratios
    df_new['BMI_Age_Ratio'] = df_new['BMI'] / df_new['Age']
    df_new['Glucose_Insulin_Ratio'] = df_new['Glucose'] / df_new['Insulin']
    
    # Catégorisation de l'âge
    df_new['Age_Category'] = pd.cut(df_new['Age'], 
                                  bins=[0, 30, 45, 60, 100],
                                  labels=['Jeune', 'Adulte', 'Moyen', 'Âgé'])
    df_new = pd.get_dummies(df_new, columns=['Age_Category'], drop_first=True)
    
    # Catégorisation du BMI
    df_new['BMI_Category'] = pd.cut(df_new['BMI'],
                                  bins=[0, 18.5, 25, 30, 100],
                                  labels=['Sous-poids', 'Normal', 'Surpoids', 'Obèse'])
    df_new = pd.get_dummies(df_new, columns=['BMI_Category'], drop_first=True)
    
    return df_new

def optimize_hyperparameters(X, y):
    """Optimise les hyperparamètres avec Optuna."""
    def objective(trial):
        # Paramètres pour Random Forest
        rf_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        
        # Paramètres pour Gradient Boosting
        gb_params = {
            'n_estimators': trial.suggest_int('gb_n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('gb_max_depth', 3, 20)
        }
        
        # Paramètres pour MLP
        mlp_params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', 
                [(64,), (32,), (64, 32), (32, 16), (64, 32, 16)]),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        }
        
        # Créer les modèles
        rf = RandomForestClassifier(**rf_params, random_state=RANDOM_STATE)
        gb = GradientBoostingClassifier(**gb_params, random_state=RANDOM_STATE)
        mlp = MLPClassifier(**mlp_params, max_iter=1000, random_state=RANDOM_STATE)
        
        # Créer l'ensemble
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('mlp', mlp)],
            voting='soft'
        )
        
        # Évaluer avec cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
        
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    return study.best_params

def perform_classification(X, y):
    """Effectue la classification avec des modèles avancés."""
    print("\n=== Classification ===")
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Créer les pipelines avec SMOTE
    pipelines = {
        'Régression Logistique': ImbPipeline([
            ('scaler', RobustScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('feature_selection', SelectKBest(f_classif, k=10)),
            ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE))
        ]),
        'Random Forest': ImbPipeline([
            ('scaler', RobustScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('feature_selection', SelectKBest(f_classif, k=10)),
            ('classifier', RandomForestClassifier(n_estimators=500, max_depth=10, random_state=RANDOM_STATE))
        ]),
        'Gradient Boosting': ImbPipeline([
            ('scaler', RobustScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('feature_selection', SelectKBest(f_classif, k=10)),
            ('classifier', GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=RANDOM_STATE))
        ]),
        'Réseau de Neurones': ImbPipeline([
            ('scaler', RobustScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('feature_selection', SelectKBest(f_classif, k=10)),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                max_iter=1000,
                alpha=0.0001,
                learning_rate_init=0.001,
                random_state=RANDOM_STATE
            ))
        ])
    }
    # Ajout XGBoost si dispo
    if XGBClassifier is not None:
        pipelines['XGBoost'] = ImbPipeline([
            ('scaler', RobustScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('feature_selection', SelectKBest(f_classif, k=10)),
            ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE))
        ])
    # Ajout LightGBM si dispo
    if LGBMClassifier is not None:
        pipelines['LightGBM'] = ImbPipeline([
            ('scaler', RobustScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('feature_selection', SelectKBest(f_classif, k=10)),
            ('classifier', LGBMClassifier(random_state=RANDOM_STATE))
        ])
    # Ajout Stacking
    estimators = []
    for name, pipe in pipelines.items():
        if name != 'Régression Logistique':
            estimators.append((name, pipe))
    if len(estimators) > 1:
        pipelines['Stacking'] = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            passthrough=True,
            n_jobs=-1
        )
    # Entraîner et évaluer les modèles
    results = {}
    for name, pipeline in pipelines.items():
        print(f"\nEntraînement du modèle: {name}")
        if name == 'Stacking':
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            if hasattr(pipeline, 'predict_proba'):
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = y_pred
        else:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        print("\nRapport de classification:")
        print(classification_report(y_test, y_pred))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        results[name] = {
            'pipeline': pipeline,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
    return results

def load_and_preprocess_data():
    """Charge et prétraite les données."""
    print("Chargement des données...")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Le fichier {DATA_PATH} n'existe pas.")
    
    # Charger les données
    df = pd.read_csv(DATA_PATH)
    
    # Afficher les informations de base
    print("\nInformations sur le dataset:")
    print(f"Nombre d'observations: {len(df)}")
    print(f"Nombre de variables: {len(df.columns)}")
    
    # Remplacer les zéros par NaN pour certaines colonnes
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_columns] = df[zero_columns].replace(0, np.nan)
    
    # Remplir les valeurs manquantes avec la médiane
    df = df.fillna(df.median())
    
    # Créer des features avancées
    df = create_advanced_features(df)
    
    print("\nNouvelles features créées:")
    print(f"Nombre total de variables: {len(df.columns)}")
    
    return df

def perform_clustering(X):
    """Effectue le clustering et trouve le nombre optimal de clusters."""
    print("\n=== Clustering ===")
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Trouver le nombre optimal de clusters
    silhouette_scores = []
    K = range(2, 11)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    
    # Trouver le meilleur k
    best_k = K[np.argmax(silhouette_scores)]
    print(f"Nombre optimal de clusters: {best_k}")
    
    # Appliquer K-means avec le meilleur k
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, kmeans, X_scaled

def visualize_results(df, clusters, classification_results):
    """Visualise les résultats de la classification et du clustering."""
    print("\n=== Visualisation des résultats ===")
    
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Courbes ROC pour tous les modèles
    plt.figure(figsize=(12, 8))
    for name, result in classification_results.items():
        plt.plot(result['fpr'], result['tpr'], 
                label=f'{name} (AUC = {result["roc_auc"]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbes ROC - Comparaison des modèles')
    plt.legend()
    plt.savefig('visualizations/roc_curves.png')
    plt.close()
    
    # 2. Analyse des clusters
    df['Cluster'] = clusters
    cluster_analysis = df.groupby('Cluster').agg({
        'Age': 'mean',
        'Glucose': 'mean',
        'BMI': 'mean',
        'Insulin': 'mean',
        'Outcome': 'mean'
    }).round(2)
    
    print("\nAnalyse des clusters:")
    print(cluster_analysis)
    
    cluster_analysis.to_csv('visualizations/cluster_analysis.csv')

def main():
    """Fonction principale."""
    try:
        # 1. Chargement et prétraitement
        df = load_and_preprocess_data()
        
        # 2. Classification
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        classification_results = perform_classification(X, y)
        
        # 3. Clustering
        clusters, kmeans, X_scaled = perform_clustering(X)
        
        # 4. Visualisation
        visualize_results(df, clusters, classification_results)
        
        print("\nAnalyse terminée avec succès!")
        print("Les visualisations ont été sauvegardées dans le dossier 'visualizations/'")
        
    except Exception as e:
        print(f"Erreur lors de l'exécution: {str(e)}")

if __name__ == "__main__":
    main() 
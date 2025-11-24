#!/usr/bin/env python3
"""
Titanic Binary Classification using Logistic Regression
========================================================

This script implements a complete binary classification pipeline to predict
survival on the Titanic using Logistic Regression.

Course: Basics of AI
Task: Binary Classification
Dataset: Titanic (from Kaggle)

Author: Wutang Repository
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def print_step(step_number, step_name):
    """Print formatted step header."""
    print("\n" + "="*80)
    print(f"STEP {step_number}: {step_name}")
    print("="*80)


def load_data(filepath='train.csv'):
    """
    Step 1: Load the Titanic dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame: Loaded dataset
    """
    print_step(1, "DATA LOADING")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Shape: {df.shape}")
        print(f"  - Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File '{filepath}' not found!")
        print("\nPlease download the Titanic dataset from:")
        print("https://www.kaggle.com/c/titanic/data")
        print("\nPlace 'train.csv' in the same directory as this script.")
        return None


def explore_data(df):
    """
    Step 2: Explore and understand the dataset.
    
    Args:
        df: Input DataFrame
    """
    print_step(2, "DATA EXPLORATION")
    
    print("\n--- Dataset Info ---")
    print(df.info())
    
    print("\n--- First Few Rows ---")
    print(df.head())
    
    print("\n--- Statistical Summary ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
    
    print("\n--- Target Variable Distribution ---")
    survival_counts = df['Survived'].value_counts()
    print(survival_counts)
    print(f"Survival Rate: {(survival_counts[1] / len(df)) * 100:.2f}%")


def visualize_data(df):
    """
    Step 3: Create visualizations to understand data distribution.
    
    Args:
        df: Input DataFrame
    """
    print_step(3, "DATA VISUALIZATION")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Survival Distribution
    df['Survived'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['#e74c3c', '#2ecc71'])
    axes[0, 0].set_title('Survival Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Survived (0 = No, 1 = Yes)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xticklabels(['Not Survived', 'Survived'], rotation=0)
    
    # 2. Age Distribution
    df['Age'].hist(bins=30, ax=axes[0, 1], color='#3498db', edgecolor='black')
    axes[0, 1].set_title('Age Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Survival by Sex
    survival_sex = df.groupby(['Sex', 'Survived']).size().unstack()
    survival_sex.plot(kind='bar', ax=axes[0, 2], color=['#e74c3c', '#2ecc71'])
    axes[0, 2].set_title('Survival by Sex', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Sex')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].legend(['Not Survived', 'Survived'])
    axes[0, 2].set_xticklabels(['Female', 'Male'], rotation=0)
    
    # 4. Survival by Pclass
    survival_pclass = df.groupby(['Pclass', 'Survived']).size().unstack()
    survival_pclass.plot(kind='bar', ax=axes[1, 0], color=['#e74c3c', '#2ecc71'])
    axes[1, 0].set_title('Survival by Passenger Class', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Pclass')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend(['Not Survived', 'Survived'])
    axes[1, 0].set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
    
    # 5. Fare Distribution
    df['Fare'].hist(bins=30, ax=axes[1, 1], color='#9b59b6', edgecolor='black')
    axes[1, 1].set_title('Fare Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Fare')
    axes[1, 1].set_ylabel('Frequency')
    
    # 6. Correlation Heatmap (numeric features only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                ax=axes[1, 2], cbar_kws={'label': 'Correlation'})
    axes[1, 2].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'data_visualization.png'")
    plt.close()


def prepare_data(df):
    """
    Step 4: Data preparation and cleaning.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame: Cleaned dataset
    """
    print_step(4, "DATA PREPARATION & CLEANING")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    print("\n--- Handling Missing Values ---")
    
    # Fill missing Age with median
    age_median = df_clean['Age'].median()
    df_clean['Age'].fillna(age_median, inplace=True)
    print(f"✓ Filled {df['Age'].isnull().sum()} missing Age values with median: {age_median:.2f}")
    
    # Fill missing Embarked with mode
    embarked_mode = df_clean['Embarked'].mode()[0]
    df_clean['Embarked'].fillna(embarked_mode, inplace=True)
    print(f"✓ Filled {df['Embarked'].isnull().sum()} missing Embarked values with mode: {embarked_mode}")
    
    # Fill missing Fare with median
    fare_median = df_clean['Fare'].median()
    df_clean['Fare'].fillna(fare_median, inplace=True)
    print(f"✓ Filled {df['Fare'].isnull().sum()} missing Fare values with median: {fare_median:.2f}")
    
    # Drop Cabin (too many missing values)
    if 'Cabin' in df_clean.columns:
        df_clean.drop('Cabin', axis=1, inplace=True)
        print("✓ Dropped 'Cabin' column (too many missing values)")
    
    print("\n--- Dropping Irrelevant Columns ---")
    # Drop columns that won't be useful for prediction
    columns_to_drop = ['PassengerId', 'Name', 'Ticket']
    for col in columns_to_drop:
        if col in df_clean.columns:
            df_clean.drop(col, axis=1, inplace=True)
            print(f"✓ Dropped '{col}' column")
    
    print(f"\n✓ Data cleaning completed. Final shape: {df_clean.shape}")
    
    return df_clean


def engineer_features(df):
    """
    Step 5: Feature engineering.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame: DataFrame with engineered features
    """
    print_step(5, "FEATURE ENGINEERING")
    
    df_feat = df.copy()
    
    # Create FamilySize feature
    df_feat['FamilySize'] = df_feat['SibSp'] + df_feat['Parch'] + 1
    print("✓ Created 'FamilySize' feature (SibSp + Parch + 1)")
    
    # Create IsAlone feature
    df_feat['IsAlone'] = (df_feat['FamilySize'] == 1).astype(int)
    print("✓ Created 'IsAlone' feature (binary indicator)")
    
    # Create Age bins
    df_feat['AgeGroup'] = pd.cut(df_feat['Age'], bins=[0, 12, 18, 35, 60, 100],
                                   labels=['Child', 'Teenager', 'Adult', 'Middle-aged', 'Senior'])
    print("✓ Created 'AgeGroup' feature (categorical age bins)")
    
    # Create Fare bins
    df_feat['FareGroup'] = pd.cut(df_feat['Fare'], bins=[0, 7.91, 14.45, 31, 1000],
                                    labels=['Low', 'Medium', 'High', 'Very High'])
    print("✓ Created 'FareGroup' feature (categorical fare bins)")
    
    # Encode categorical variables
    print("\n--- Encoding Categorical Variables ---")
    
    # Sex: Male=1, Female=0
    df_feat['Sex'] = df_feat['Sex'].map({'male': 1, 'female': 0})
    print("✓ Encoded 'Sex': male=1, female=0")
    
    # Embarked: One-hot encoding
    embarked_dummies = pd.get_dummies(df_feat['Embarked'], prefix='Embarked')
    df_feat = pd.concat([df_feat, embarked_dummies], axis=1)
    df_feat.drop('Embarked', axis=1, inplace=True)
    print(f"✓ One-hot encoded 'Embarked': {list(embarked_dummies.columns)}")
    
    # AgeGroup: One-hot encoding
    agegroup_dummies = pd.get_dummies(df_feat['AgeGroup'], prefix='AgeGroup')
    df_feat = pd.concat([df_feat, agegroup_dummies], axis=1)
    df_feat.drop('AgeGroup', axis=1, inplace=True)
    print(f"✓ One-hot encoded 'AgeGroup': {list(agegroup_dummies.columns)}")
    
    # FareGroup: One-hot encoding
    faregroup_dummies = pd.get_dummies(df_feat['FareGroup'], prefix='FareGroup')
    df_feat = pd.concat([df_feat, faregroup_dummies], axis=1)
    df_feat.drop('FareGroup', axis=1, inplace=True)
    print(f"✓ One-hot encoded 'FareGroup': {list(faregroup_dummies.columns)}")
    
    print(f"\n✓ Feature engineering completed. Final shape: {df_feat.shape}")
    
    return df_feat


def split_data(df, test_size=0.2, random_state=42):
    """
    Step 6: Split data into training and testing sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of dataset to include in test split
        random_state: Random seed
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print_step(6, "TRAIN/TEST SPLIT")
    
    # Separate features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Data split completed:")
    print(f"  - Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"  - Testing set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    print(f"  - Number of features: {X_train.shape[1]}")
    print(f"\n  - Training set survival rate: {(y_train.sum() / len(y_train)) * 100:.2f}%")
    print(f"  - Testing set survival rate: {(y_test.sum() / len(y_test)) * 100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    """
    Step 7: Feature normalization using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Testing features
        
    Returns:
        tuple: X_train_scaled, X_test_scaled, scaler
    """
    print_step(7, "FEATURE NORMALIZATION")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Features normalized using StandardScaler")
    print(f"  - Mean: {X_train_scaled.mean():.6f}")
    print(f"  - Std: {X_train_scaled.std():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):
    """
    Step 8: Train Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        LogisticRegression: Trained model
    """
    print_step(8, "MODEL TRAINING")
    
    # Initialize and train the model
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    )
    
    print("Training Logistic Regression model...")
    model.fit(X_train, y_train)
    
    print("✓ Model trained successfully!")
    print(f"  - Algorithm: Logistic Regression")
    print(f"  - Solver: {model.solver}")
    print(f"  - Max iterations: {model.max_iter}")
    print(f"  - Number of iterations: {model.n_iter_[0]}")
    
    return model


def cross_validate_model(model, X_train, y_train, cv=5):
    """
    Step 9: Perform cross-validation.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        cv: Number of folds
        
    Returns:
        array: Cross-validation scores
    """
    print_step(9, "CROSS-VALIDATION")
    
    print(f"Performing {cv}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"✓ Cross-validation completed!")
    print(f"  - Fold scores: {cv_scores}")
    print(f"  - Mean accuracy: {cv_scores.mean():.4f}")
    print(f"  - Standard deviation: {cv_scores.std():.4f}")
    print(f"  - Min accuracy: {cv_scores.min():.4f}")
    print(f"  - Max accuracy: {cv_scores.max():.4f}")
    
    return cv_scores


def hyperparameter_tuning(X_train, y_train):
    """
    Step 10: Hyperparameter tuning with GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        LogisticRegression: Best model from grid search
    """
    print_step(10, "HYPERPARAMETER TUNING")
    
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  - {param}: {values}")
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    print("\nPerforming grid search...")
    grid_search.fit(X_train, y_train)
    
    print("✓ Grid search completed!")
    print(f"\n  Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    - {param}: {value}")
    print(f"\n  Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Step 11: Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print_step(11, "MODEL EVALUATION")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    print("\n--- Performance Metrics ---")
    print(f"Training Accuracy:   {metrics['train_accuracy']:.4f}")
    print(f"Testing Accuracy:    {metrics['test_accuracy']:.4f}")
    print(f"Precision:           {metrics['precision']:.4f}")
    print(f"Recall:              {metrics['recall']:.4f}")
    print(f"F1-Score:            {metrics['f1_score']:.4f}")
    print(f"ROC-AUC Score:       {metrics['roc_auc']:.4f}")
    
    # Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"\nTrue Negatives:  {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives:  {cm[1, 1]}")
    
    # Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Not Survived', 'Survived']))
    
    return metrics, y_test_pred, y_test_proba


def visualize_results(model, X_train, X_test, y_train, y_test, 
                      y_test_pred, y_test_proba, feature_names):
    """
    Step 12: Create comprehensive visualizations of results.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        y_test_pred: Predictions on test set
        y_test_proba: Prediction probabilities on test set
        feature_names: Names of features
    """
    print_step(12, "RESULTS VISUALIZATION")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xlabel('Predicted')
    
    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    axes[0, 1].plot(fpr, tpr, color='#2ecc71', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], color='#e74c3c', lw=2, linestyle='--', 
                    label='Random Classifier')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature Importance (Coefficients)
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False).head(10)
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in coefficients['Coefficient']]
    axes[1, 0].barh(coefficients['Feature'], coefficients['Coefficient'], color=colors)
    axes[1, 0].set_xlabel('Coefficient Value')
    axes[1, 0].set_title('Top 10 Feature Coefficients', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 4. Prediction Distribution
    axes[1, 1].hist(y_test_proba, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0.5, color='#e74c3c', linestyle='--', linewidth=2, 
                       label='Decision Threshold (0.5)')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Predicted Probabilities', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    print("✓ Evaluation visualization saved as 'model_evaluation.png'")
    plt.close()
    
    # Save feature coefficients to CSV
    all_coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)
    
    all_coefficients.to_csv('feature_coefficients.csv', index=False)
    print("✓ Feature coefficients saved as 'feature_coefficients.csv'")


def save_results(metrics, cv_scores):
    """
    Save final results to a text file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        cv_scores: Cross-validation scores
    """
    with open('model_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("TITANIC SURVIVAL PREDICTION - LOGISTIC REGRESSION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("--- Cross-Validation Results ---\n")
        f.write(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
        f.write(f"Fold Scores: {cv_scores}\n\n")
        
        f.write("--- Final Model Performance ---\n")
        f.write(f"Training Accuracy:   {metrics['train_accuracy']:.4f}\n")
        f.write(f"Testing Accuracy:    {metrics['test_accuracy']:.4f}\n")
        f.write(f"Precision:           {metrics['precision']:.4f}\n")
        f.write(f"Recall:              {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:            {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC Score:       {metrics['roc_auc']:.4f}\n")
        
    print("\n✓ Results saved to 'model_results.txt'")


def main():
    """
    Main function to execute the complete binary classification pipeline.
    """
    print("\n" + "="*80)
    print("TITANIC SURVIVAL PREDICTION USING LOGISTIC REGRESSION")
    print("="*80)
    print("Course: Basics of AI")
    print("Task: Binary Classification")
    print("Dataset: Titanic (Kaggle)")
    print("="*80)
    
    # Step 1: Load Data
    df = load_data('train.csv')
    if df is None:
        return
    
    # Step 2: Explore Data
    explore_data(df)
    
    # Step 3: Visualize Data
    visualize_data(df)
    
    # Step 4: Prepare Data
    df_clean = prepare_data(df)
    
    # Step 5: Engineer Features
    df_feat = engineer_features(df_clean)
    
    # Step 6: Split Data
    X_train, X_test, y_train, y_test = split_data(df_feat)
    
    # Step 7: Normalize Features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # Store feature names for later use
    feature_names = X_train.columns.tolist()
    
    # Step 8: Train Model
    model = train_model(X_train_scaled, y_train)
    
    # Step 9: Cross-Validation
    cv_scores = cross_validate_model(model, X_train_scaled, y_train, cv=5)
    
    # Step 10: Hyperparameter Tuning
    best_model = hyperparameter_tuning(X_train_scaled, y_train)
    
    # Step 11: Evaluate Model
    metrics, y_test_pred, y_test_proba = evaluate_model(
        best_model, X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Step 12: Visualize Results
    visualize_results(
        best_model, X_train_scaled, X_test_scaled, y_train, y_test,
        y_test_pred, y_test_proba, feature_names
    )
    
    # Save Results
    save_results(metrics, cv_scores)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Files:")
    print("  1. data_visualization.png - Data exploration visualizations")
    print("  2. model_evaluation.png - Model performance visualizations")
    print("  3. feature_coefficients.csv - Feature importance rankings")
    print("  4. model_results.txt - Summary of model performance")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import pprint
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pickle


with open('df_cleaned.pkl', 'rb') as f:
    df_cleaned = pickle.load(f)

X, y_dgf, y_crr2_lt30, y_crr2 = df_cleaned.drop(columns=["crr2_lt30", "dgf", "crr2"]), df_cleaned['dgf'], df_cleaned['crr2_lt30'], df_cleaned['crr2']

normalized_vars = ['rage', 'rbmi', 'tdialyr', 'dage', 'dbmi', 'cit']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical columns
X[normalized_vars] = scaler.fit_transform(X[normalized_vars])


model_df = {}

def model_evaluation(model, X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fitting the model
    model.fit(X_train, y_train)

    # Predicting the test set
    y_pred = model.predict(X_test)

    # Predict probabilities for AUC score
    y_prob = model.predict_proba(X_test)[:, 1]  # Assuming binary classification

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculating AUC score
    auc = roc_auc_score(y_test, y_prob)

    # Calculating F1 scores for both classes
    f1_class1 = f1_score(y_test, y_pred, pos_label=1)  # For Class 1
    f1_class2 = f1_score(y_test, y_pred, pos_label=0)  # For Class 2

    # Cross-validation score
    cross_val_avg = np.mean(cross_val_score(model, X, y, cv=5))

    # Printing out the results
    print(f"Model: {model}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1 Score (Class 1): {f1_class1:.4f}")
    print(f"F1 Score (Class 2): {f1_class2:.4f}")
    print(f"Average Cross-validation Score: {cross_val_avg:.4f}")

    # Storing the cross-validation score in a dictionary
    model_df[model] = {
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'F1 Score Class 1': round(f1_class1, 4),
        'F1 Score Class 2': round(f1_class2, 4),
        'Cross-validation Score': round(cross_val_avg, 4)
    }

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, make_scorer
import pandas as pd
import numpy as np

# Random states, criterion, n_estimators, max_depth to search
param_grid = {
    'random_state': [32, 100, 777, 2022, 8888],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'n_estimators': [8, 16, 50, 100, 150, 200],
    'max_depth': [2, 4, 8, 16, 32]
}

# Custom scorer for specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Custom scorer for weighted specificity
def weighted_specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    specificity = tn / (tn + fp)
    return specificity

# Define scoring dictionary
scoring = {
    'AUC': 'roc_auc',
    'Accuracy': 'accuracy',
    'F1_weighted': make_scorer(f1_score, average='weighted'),
    'Sensitivity': make_scorer(recall_score, pos_label=1),
    'Specificity': make_scorer(specificity_score),
    'Precision': make_scorer(precision_score, pos_label=1, zero_division=0),
    'Recall': make_scorer(recall_score, pos_label=1),
    'Sensitivity_weighted': make_scorer(recall_score, average='weighted'),
    'Specificity_weighted': make_scorer(weighted_specificity_score),
    'Precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
    'Recall_weighted': make_scorer(recall_score, average='weighted')
}

def calculate_confidence_interval(data):
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    ci = 1.96 * std_err
    return mean, ci

def run_grid_search(X, y):
    # Set up the Random Forest Classifier and GridSearchCV
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scoring, cv=5, n_jobs=-1, return_train_score=True, refit='Accuracy')
    
    # Fit the grid search on the entire dataset
    grid_search.fit(X, y)
    
    # Access all the results
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Sort the models by mean test accuracy score
    results = results.sort_values(by='mean_test_AUC', ascending=False)
    
    # Get top N models by mean test accuracy score
    top_models = results
    
    top_models_metrics = []
    
    for i, row in top_models.iterrows():
        params = {key: row['param_' + key] for key in param_grid}
        
        # Collect mean metrics and confidence intervals from cross-validation results
        metrics = {
            'AUC': f"{round(row['mean_test_AUC'], 4)} ± {round(calculate_confidence_interval([row[f'split{j}_test_AUC'] for j in range(5)])[1], 4)}",
            'Accuracy': f"{round(row['mean_test_Accuracy'], 4)} ± {round(calculate_confidence_interval([row[f'split{j}_test_Accuracy'] for j in range(5)])[1], 4)}",
            'F1_weighted': f"{round(row['mean_test_F1_weighted'], 4)} ± {round(calculate_confidence_interval([row[f'split{j}_test_F1_weighted'] for j in range(5)])[1], 4)}",
            'Sensitivity': f"{round(row['mean_test_Sensitivity'], 4)} ± {round(calculate_confidence_interval([row[f'split{j}_test_Sensitivity'] for j in range(5)])[1], 4)}",
            'Specificity': f"{round(row['mean_test_Specificity'], 4)} ± {round(calculate_confidence_interval([row[f'split{j}_test_Specificity'] for j in range(5)])[1], 4)}",
            'Precision': f"{round(row['mean_test_Precision'], 4)} ± {round(calculate_confidence_interval([row[f'split{j}_test_Precision'] for j in range(5)])[1], 4)}",
            'Recall': f"{round(row['mean_test_Recall'], 4)} ± {round(calculate_confidence_interval([row[f'split{j}_test_Recall'] for j in range(5)])[1], 4)}",
            'Sensitivity_weighted': f"{round(row['mean_test_Sensitivity_weighted'], 4)} ± {round(calculate_confidence_interval([row[f'split{j}_test_Sensitivity_weighted'] for j in range(5)])[1], 4)}",
            'Specificity_weighted': f"{round(row['mean_test_Specificity_weighted'], 4)} ± {round(calculate_confidence_interval([row[f'split{j}_test_Specificity_weighted'] for j in range(5)])[1], 4)}",
            'Precision_weighted': f"{round(row['mean_test_Precision_weighted'], 4)} ± {round(calculate_confidence_interval([row[f'split{j}_test_Precision_weighted'] for j in range(5)])[1], 4)}",
            'Recall_weighted': f"{round(row['mean_test_Recall_weighted'], 4)} ± {round(calculate_confidence_interval([row[f'split{j}_test_Recall_weighted'] for j in range(5)])[1], 4)}",
            'params': params,
        }
        
        top_models_metrics.append(metrics)
    
    # Save the best model by AUC
    with open('best_rf_model.pkl', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    
    return pd.DataFrame(top_models_metrics), grid_search.best_estimator_

if __name__ == "__main__":
    # Example: run grid search for DGF outcome
    print("Running grid search for DGF outcome...")
    top_models_df, best_model = run_grid_search(X, y_dgf)
    print(top_models_df.head())
    print("Best model saved as best_rf_model.pkl")

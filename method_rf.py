from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

def scale():
    # Load dataset
    df = pd.read_csv("polymarket_ml.csv")

    df = df.dropna()
    print(df["category"].value_counts()) # how many markets are in each category after dropping na

    df = df.drop(columns=["market_spread"]) #this column is not useful for classification

    feature_cols = [col for col in df.columns if col != "category"]
    X = df[feature_cols]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Test set size: {len(y_test)}")

    # for random forest, we don't need to scale the data

    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = scale()
    
    param_grid = { 
        'n_estimators': [100, 200], 
        'max_depth': [None, 4, 8], 
        'min_samples_leaf': [1, 2, 5], 
        'max_features': ['sqrt', None] 
    }

    base_model = RandomForestClassifier(random_state=42) 

    grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,                                # 5-fold cross-validation
    scoring='accuracy', # loss function 
    n_jobs=-1                            
    )

    grid_search.fit(X_train, y_train) # fit the model to the training data

    best_rf = grid_search.best_estimator_

    print(f"Optimal Parameters: {grid_search.best_params_}") # print the best parameters
    print(f"Best 5-Fold CV Accuracy: {grid_search.best_score_ * 100:.2f}%") # print the best validation accuracy

    test_preds = best_rf.predict(X_test) # predict the test data
    train_preds = best_rf.predict(X_train) # predict the training data

    print(f"Training Accuracy (Best Model): {accuracy_score(y_train, train_preds) * 100:.2f}%") # print the training accuracy
    print(f"Test Set Accuracy:              {accuracy_score(y_test, test_preds) * 100:.2f}%\n") # print the test accuracy
    print(classification_report(y_test, test_preds)) # print the classification report

    importances = pd.Series(best_rf.feature_importances_, index=X_train.columns) # print the feature importance
    print("--- Feature Importance Ranking ---")
    print((importances.sort_values(ascending=False) * 100).round(2).astype(str) + '%') # print the feature importance

if __name__ == "__main__":
    main()

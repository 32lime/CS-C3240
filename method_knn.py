from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

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

    # These features are heavily right-skewed so we use log to compress them
    log_cols = [
        "lifespan_days",
        "market_total_volume",
        "trade_size_avg_usd_last_2500",
        "trade_size_std_usd_last_2500",
        "trades_per_hour_last_2500_trades",
    ]

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[log_cols] = np.log1p(X_train[log_cols])
    X_test[log_cols] = np.log1p(X_test[log_cols])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def main():
    X_train_scaled, X_test_scaled, y_train, y_test = scale()

    param_grid = {"n_neighbors": [1, 3, 5, 7, 9, 11]} # number of neighbors to consider when choosing

    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring="accuracy",
    )

    grid_search.fit(X_train_scaled, y_train)

    for k, mean_score in zip(
        grid_search.cv_results_["param_n_neighbors"],
        grid_search.cv_results_["mean_test_score"],
    ):
        print(f"k={k:<2} | CV Accuracy: {mean_score:.4f}")

    best_knn = grid_search.best_estimator_

    print(f"Optimal Parameters: {grid_search.best_params_}")
    print(f"Best 5-Fold CV Accuracy: {grid_search.best_score_ * 100:.2f}%")

    train_preds = best_knn.predict(X_train_scaled)
    test_preds = best_knn.predict(X_test_scaled)

    print(f"Training Accuracy (Best Model): {accuracy_score(y_train, train_preds) * 100:.2f}%")
    print(f"Test Set Accuracy:              {accuracy_score(y_test, test_preds) * 100:.2f}%")


if __name__ == "__main__":
    main()

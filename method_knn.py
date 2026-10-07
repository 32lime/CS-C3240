from sklearn.model_selection import train_test_split
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

    best_k = 0
    best_score = 0

    for k in [1, 3, 5, 7, 9, 11]:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train_scaled, y_train)
        y_prediction = knn_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_prediction)

        y_train_prediction = knn_model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_prediction)
        
        print(f"k={k:<2} | Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {accuracy:.4f}")

        if accuracy > best_score:
            best_score = accuracy
            best_k = k
    print(f"Best k: {best_k} with accuracy: {best_score}")


if __name__ == "__main__":
    main()

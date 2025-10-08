import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save(csv_path, model_path):
    # Load real leads.csv
    df = pd.read_csv(csv_path)

    # Select features (non-text)
    features = ["lead_score", "company_size", "industry", "persona", "region", "preferred_channel"]
    X = df[features].copy()

    # Handle missing values
    X = X.fillna("Unknown")

    # Encode categorical columns
    encoders = {}
    for col in ["company_size", "industry", "persona", "region", "preferred_channel"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Label encode target
    y = df["triage_category"]
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("Train accuracy:", clf.score(X_train, y_train))
    print("Test accuracy:", clf.score(X_test, y_test))

    # Save model + encoders
    joblib.dump({"model": clf, "encoders": encoders, "label_encoder": y_le}, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save(sys.argv[1], sys.argv[2])

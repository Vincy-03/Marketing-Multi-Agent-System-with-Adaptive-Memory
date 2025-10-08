import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json, os

MODEL_PATH = "models/lead_model.pkl"  # or balanced model later
DATA_CSV = "data/marketing_multi_agent_dataset_v1_final/leads.csv"

print("Loading model and data...")
m = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_CSV)

# Same features as training
features = ["lead_score", "company_size", "industry", "persona", "region", "preferred_channel"]
X = df[features].fillna("Unknown")

# Apply encoders if they exist
if isinstance(m, dict):
    model = m["model"]
    encoders = m.get("encoders", {})
    for col, le in encoders.items():
        X[col] = le.transform(X[col].astype(str))
    y_le = m["label_encoder"]
    y = y_le.transform(df["triage_category"].astype(str))
else:
    model = m
    y = df["triage_category"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Predictions
y_pred = model.predict(X_test)

print("\nClassification report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save report for later
os.makedirs("reports", exist_ok=True)
with open("reports/diag_report.json", "w") as f:
    json.dump({
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }, f, indent=2)

print("\nSaved detailed report to reports/diag_report.json")

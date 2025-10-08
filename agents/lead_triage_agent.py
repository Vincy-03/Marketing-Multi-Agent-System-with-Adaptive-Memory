# agents/lead_triage_agent.py
from fastapi import APIRouter
from pydantic import BaseModel
import joblib, os, pandas as pd
from typing import Optional, Dict, Any

router = APIRouter()

# Input schema: either provide lead_id (preferred) or explicit features to classify
class LeadInput(BaseModel):
    lead_id: Optional[str] = None
    # optional feature overrides (if you want to test a custom example)
    lead_score: Optional[float] = None
    company_size: Optional[str] = None
    industry: Optional[str] = None
    persona: Optional[str] = None
    region: Optional[str] = None
    preferred_channel: Optional[str] = None

# Paths
MODEL_PATH = os.path.join("models", "lead_model.pkl")
LEADS_CSV = os.path.join("data", "marketing_multi_agent_dataset_v1_final", "leads.csv")

# Load model at import time (if available)
model_bundle = None
if os.path.exists(MODEL_PATH):
    model_bundle = joblib.load(MODEL_PATH)
    if isinstance(model_bundle, dict):
        clf = model_bundle.get("model")
        encoders = model_bundle.get("encoders", {})
        label_encoder = model_bundle.get("label_encoder", None)
    else:
        clf = model_bundle
        encoders = {}
        label_encoder = None
else:
    clf = None
    encoders = {}
    label_encoder = None

def lookup_lead(lead_id: str) -> Dict[str, Any]:
    """Return a dict row for lead_id from leads.csv or empty dict if not found."""
    if not os.path.exists(LEADS_CSV):
        return {}
    try:
        df = pd.read_csv(LEADS_CSV)
    except Exception:
        return {}
    row = df[df['lead_id'] == lead_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()

def build_feature_df(source: Dict[str, Any], overrides: Dict[str, Any]):
    features = ["lead_score", "company_size", "industry", "persona", "region", "preferred_channel"]
    out = {f: "Unknown" for f in features}
    for k, v in source.items():
        if k in out and pd.notna(v):
            out[k] = v
    for k, v in overrides.items():
        if k in out and v is not None:
            out[k] = v
    try:
        out["lead_score"] = float(out.get("lead_score") if out.get("lead_score") not in (None, "") else 0.0)
    except Exception:
        out["lead_score"] = 0.0
    return pd.DataFrame([out])

def map_probs_to_labels(probs):
    """Return an ordered dict mapping label -> probability using the saved label_encoder if available."""
    result = {}
    if label_encoder is not None:
        try:
            classes = list(label_encoder.classes_)
            for i, p in enumerate(probs):
                # classes[i] exists for valid index
                lbl = classes[i] if i < len(classes) else str(i)
                result[str(lbl)] = float(p)
            return result
        except Exception:
            pass
    # fallback: return index->prob mapping if no label encoder
    for i, p in enumerate(probs):
        result[str(i)] = float(p)
    return result

@router.post("/triage")
async def triage(input: LeadInput):
    if clf is None:
        return {"error": "Model not available. Please train the model and create models/lead_model.pkl"}

    source_row = {}
    if input.lead_id:
        source_row = lookup_lead(input.lead_id)
        if not source_row:
            msg = f"Lead id {input.lead_id} not found in leads.csv; using provided overrides if any."
        else:
            msg = f"Found lead {input.lead_id} in leads.csv."
    else:
        msg = "No lead_id provided; using provided overrides if any."

    overrides = {
        "lead_score": input.lead_score,
        "company_size": input.company_size,
        "industry": input.industry,
        "persona": input.persona,
        "region": input.region,
        "preferred_channel": input.preferred_channel,
    }
    X_df = build_feature_df(source_row, overrides)

    # Apply encoders if present
    X_enc = X_df.copy()
    for col, le in encoders.items():
        if col in X_enc.columns:
            try:
                X_enc[col] = X_enc[col].astype(str).map(
                    lambda v: v if v in le.classes_ else "Unknown"
                )
                if "Unknown" not in list(le.classes_):
                    classes = list(le.classes_) + ["Unknown"]
                    mapping = {c: i for i, c in enumerate(classes)}
                    X_enc[col] = X_enc[col].map(mapping)
                else:
                    X_enc[col] = le.transform(X_enc[col].astype(str))
            except Exception:
                X_enc[col] = X_enc[col].astype(str)

    feature_order = ["lead_score", "company_size", "industry", "persona", "region", "preferred_channel"]
    X_final = X_enc[feature_order]

    try:
        pred_idx = clf.predict(X_final)[0]
        probs = clf.predict_proba(X_final)[0].tolist() if hasattr(clf, "predict_proba") else None

        if label_encoder is not None:
            try:
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
            except Exception:
                pred_label = str(pred_idx)
        else:
            pred_label = str(pred_idx)

        probs_map = map_probs_to_labels(probs) if probs is not None else None

    except Exception as e:
        return {"error": "Prediction failed", "detail": str(e)}

    response = {
        "lead_id": input.lead_id,
        "predicted_category": pred_label,
        "message": msg
    }
    if probs_map is not None:
        response["class_probabilities"] = probs_map
    elif probs is not None:
        response["class_probabilities_raw"] = probs

    return response

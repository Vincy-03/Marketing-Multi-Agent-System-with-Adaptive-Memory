# memory/long_term.py
import json, os

STORE_PATH = os.path.join("memory", "long_term_store.json")

def load_long_term():
    if os.path.exists(STORE_PATH):
        with open(STORE_PATH, "r", encoding="utf8") as f:
            return json.load(f)
    return {}

def save_long_term(data):
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2)

# Simple getter
def get_lead_preferences(lead_id: str):
    data = load_long_term()
    return data.get(lead_id, {})

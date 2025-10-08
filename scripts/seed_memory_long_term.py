# scripts/seed_memory_long_term.py
import pandas as pd
import json, os

DATA_PATH = os.path.join("data", "marketing_multi_agent_dataset_v1_final", "memory_long_term.csv")
OUT_PATH = os.path.join("memory", "long_term_store.json")

def seed_long_term():
    if not os.path.exists(DATA_PATH):
        print("ERROR: memory_long_term.csv not found at", DATA_PATH)
        return
    
    # Load CSV
    df = pd.read_csv(DATA_PATH)
    print("Loaded memory_long_term.csv:", df.shape)

    # Expected columns: lead_id, region, industry, rfm_score, preferences_json, last_updated_at
    store = {}
    for _, row in df.iterrows():
        lead_id = str(row['lead_id'])
        prefs = {}
        prefs['region'] = row.get('region', "")
        prefs['industry'] = row.get('industry', "")
        prefs['rfm_score'] = row.get('rfm_score', "")
        # preferences_json is already JSON string
        try:
            prefs_data = json.loads(row['preferences_json']) if pd.notna(row['preferences_json']) else {}
            prefs.update(prefs_data)
        except Exception as e:
            print(f"Warning: could not parse preferences for {lead_id}: {e}")
        store[lead_id] = prefs

    # Save to JSON file
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf8") as f:
        json.dump(store, f, indent=2)

    print(f"✅ Seeded {len(store)} records into {OUT_PATH}")

if __name__ == "__main__":
    seed_long_term()

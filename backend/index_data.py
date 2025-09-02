import os
import json
import time
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_FILE = os.getenv("DATA_FILE", os.path.join(os.path.dirname(__file__), "sample_data.jsonl"))

def wait_api_ready(url: str, retries: int = 30, delay: int = 3):
    for i in range(retries):
        try:
            r = requests.get(f"{url}/docs", timeout=3)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        print(f"[index_data] API not ready yet (try {i+1}/{retries}); waiting {delay}s...")
        time.sleep(delay)
    return False

def main():
    if not wait_api_ready(API_URL):
        print("[index_data] API was not reachable in time. Aborting.")
        return

    # Read JSONL sample data
    if not os.path.exists(DATA_FILE):
        print(f"[index_data] Data file not found: {DATA_FILE}")
        return

    indexed = 0
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            resp = requests.post(f"{API_URL}/index-product", json=obj, timeout=60)
            if resp.status_code == 201:
                indexed += 1
                print(f"[index_data] Indexed: {obj.get('product_id')}")
            else:
                print(f"[index_data] Failed: {obj.get('product_id')} -> {resp.status_code} {resp.text}")
    print(f"[index_data] Done. Indexed {indexed} products.")

if __name__ == "__main__":
    main()

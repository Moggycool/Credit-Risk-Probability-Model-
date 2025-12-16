# post_payload.py
import json
import requests
with open("payload.json", "r", encoding="utf-8") as fh:
    payload = json.load(fh)
r = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
print("status:", r.status_code)
print("body:", r.text)
with open("logs/response_smoke_test_1.json", "w", encoding="utf-8") as fh:
    fh.write(r.text)

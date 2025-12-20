"""Manual tests for the API endpoints."""
import requests
import json


def test_api_manually():
    """Manual test of the API endpoints."""

    base_url = "http://localhost:8000"

    print("1. Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    print()

    print("2. Testing prediction endpoint...")
    payload = {
        "customer_id": "MANUAL_TEST_001",
        "features": {
            "Year_mean": 2019.0,
            "Month_mean": 8.0
        }
    }

    response = requests.post(f"{base_url}/predict", json=payload)
    print(f"   Status: {response.status_code}")

    if response.status_code == 200:
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"   Error: {response.text}")

    print()
    print("3. Testing with wrong features...")
    wrong_payload = {
        "customer_id": "WRONG_TEST",
        "features": {
            "Wrong_feature": 100.0,
            "Another_wrong": 50.0
        }
    }

    response = requests.post(f"{base_url}/predict", json=wrong_payload)
    print(f"   Status: {response.status_code}")
    print(f"   Error: {response.text}")


if __name__ == "__main__":
    test_api_manually()

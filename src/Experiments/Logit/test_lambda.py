import os
import requests

API_URL = "https://cloud.lambda.ai/api/v1/instances"


api_key = os.getenv("LAMBDA_API_KEY")
if not api_key:
    raise SystemExit("❌ Missing environment variable: LAMBDA_API_KEY")

headers = {"Authorization": f"Bearer {api_key}"}

payload = {
    "region_name": "us-west-2",
    "instance_type_name": "gpu_1x_a100_80gb",
    "ssh_key_names": ["algo25"]
}

print("🚀 Sending test request to Lambda Cloud API...")
resp = requests.post(API_URL, headers=headers, json=payload)

print("📌 HTTP Status:", resp.status_code)
print("📬 Response Text (first 300 chars):")
print(resp.text[:300])

try:
    data = resp.json()
    print("\n🔍 Parsed JSON:")
    print(data)
except Exception:
    print("\n⚠️ Could not decode JSON response.")

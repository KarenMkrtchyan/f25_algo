import os
import requests

BASE_URL = "https://cloud.lambdalabs.com/api"

api_key = os.getenv("LAMBDA_API_KEY")
if not api_key:
    raise SystemExit("❌ Missing LAMBDA_API_KEY env variable")

HEADERS = {"Authorization": f"Bearer {api_key}"}

CANDIDATE_ENDPOINTS = [
    "/v1/instances",
    "/v1/instance-types",
    "/v1/regions",
    "/v1/ssh-keys",
    "/v1/billing",
    "/v1/account",
]

METHODS = ["GET", "POST", "DELETE"]

print("🔍 Testing Lambda API endpoints...\n")

for ep in CANDIDATE_ENDPOINTS:
    url = BASE_URL + ep
    print(f"=== {url} ===")
    for method in METHODS:
        try:
            resp = requests.request(method, url, headers=HEADERS)
            body = resp.text[:200].replace("\n", " ")
            print(f"{method:6} -> {resp.status_code:3} | {body}")
        except Exception as e:
            print(f"{method:6} -> ERROR {e}")
    print()

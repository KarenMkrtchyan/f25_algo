import os, requests

BASE = "https://cloud.lambdalabs.com/api/v1"
api_key = os.getenv("LAMBDA_API_KEY")
assert api_key, "Missing LAMBDA_API_KEY"
H = {"Authorization": f"Bearer {api_key}"}

ENDPOINTS = [
    "/instances/operations/launch",
    "/instance-operations/launch",
    "/operations/launch",
    "/instances/actions",
]

METHODS = ["GET", "POST"]

print("🔍 Searching for instance launch endpoint...\n")

for ep in ENDPOINTS:
    url = BASE + ep
    print(f"== {url} ==")
    for m in METHODS:
        try:
            r = requests.request(m, url, headers=H)
            print(f"{m:5} -> {r.status_code} | {r.text[:200]}")
        except Exception as e:
            print(f"{m:5} -> {e}")
    print()

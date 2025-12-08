import os, requests

BASE = "https://cloud.lambdalabs.com/api/v1"
api_key = os.getenv("LAMBDA_API_KEY")
assert api_key, "Missing LAMBDA_API_KEY"
H = {"Authorization": f"Bearer {api_key}"}

print("\n📌 Fetching GPU types & regions...\n")
resp = requests.get(f"{BASE}/instance-types", headers=H).json()

for name, info in resp["data"].items():
    print(f"🖥 {name}")
    print(f" • GPU: {info['instance_type']['gpu_description']}")
    print(f" • Regions:")
    for r in info["regions_with_capacity_available"]:
        print("    -", r)
    print()

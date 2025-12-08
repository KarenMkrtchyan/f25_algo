import os
import time
import subprocess
import requests
from typing import Dict, Any

BASE_URL = "https://cloud.lambdalabs.com/api/v1"

def headers():
    api_key = os.getenv("LAMBDA_API_KEY")
    if not api_key:
        raise RuntimeError("⚠️ Missing env var: LAMBDA_API_KEY")
    return {"Authorization": f"Bearer {api_key}"}

def wait_for_instance_ready(instance_id, timeout=600):
    url = f"{BASE_URL}/instances/{instance_id}"
    print(f"⏳ Waiting for instance {instance_id} to become active...")

    for i in range(timeout // 10):
        resp = requests.get(url, headers=headers())
        if resp.status_code != 200:
            print("⚠️ Poll failed, retrying...")
            time.sleep(10)
            continue

        data = resp.json()
        status = data["data"]["status"]
        print(f"   → Status: {status}")

        if status == "active":
            ip = data["data"]["ip"]
            print(f"🎉 Instance is ACTIVE with IP: {ip}")
            return ip
        
        time.sleep(10)

    raise TimeoutError("❌ Instance did not become active in time")

def launch_instance(instance_type, region_name, ssh_key_id, ssh_key_name):
    url = f"{BASE_URL}/instance-operations/launch"

    payload = {
        "instance_type_name": instance_type,
        "region_name": region_name,
        "ssh_key_ids": [ssh_key_id],
        "ssh_key_names": [ssh_key_name]
    }

    resp = requests.post(url, headers=headers(), json=payload)
    print("🔥 Launch response status:", resp.status_code)
    print("🔥 Launch response body:", resp.text)
    resp.raise_for_status()

    data = resp.json()
    instance_id = data["data"]["instance_ids"][0]
    print("🚀 Launched instance:", instance_id)

    return instance_id



def wait_for_operation(op_id: str, poll_sec: int = 10) -> str:
    """
    Poll the launch operation until it finishes and return the instance id.
    """
    url = f"{BASE_URL}/instance-operations/{op_id}"
    while True:
        resp = requests.get(url, headers=headers())
        if resp.status_code != 200:
            raise RuntimeError(
                f"Polling operation failed ({resp.status_code}): {resp.text}"
            )
        data = resp.json()["data"]
        status = data["status"]
        print(f"⏳ Operation status: {status}")
        if status == "completed":
            instance_id = data["result"]["instance_id"]
            print(f"✅ Operation completed, instance id: {instance_id}")
            return instance_id
        if status in ("failed", "canceled"):
            raise RuntimeError(f"Operation ended with status={status}: {data}")
        time.sleep(poll_sec)


def get_instance_ip(instance_id: str) -> str:
    """
    Get the public IP of an instance by id.
    """
    url = f"{BASE_URL}/instances/{instance_id}"
    resp = requests.get(url, headers=headers())
    if resp.status_code != 200:
        raise RuntimeError(
            f"Fetching instance failed ({resp.status_code}): {resp.text}"
        )
    data = resp.json()["data"]
    status = data["status"]
    ip = data["ip"]
    print(f"🖥 Instance status={status}, ip={ip}")
    if status != "active":
        raise RuntimeError(f"Instance not active yet (status={status})")
    if not ip:
        raise RuntimeError("Instance has no public IP yet")
    return ip


def ssh_and_run(ip: str, remote_cmd: str, ssh_key_path: str = None):
    """
    SSH into the instance and run a shell command.
    Assumes username 'ubuntu' (Lambda default images).
    """
    user_at_host = f"ubuntu@{ip}"
    key_arg = []
    if ssh_key_path:
        key_arg = ["-i", ssh_key_path]

    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        *key_arg,
        user_at_host,
        remote_cmd,
    ]
    print("🔧 Running SSH command:")
    print(" ", " ".join(cmd))
    subprocess.check_call(cmd)


def terminate_instance(instance_id: str):
    """
    Terminate an instance to stop billing.
    """
    url = f"{BASE_URL}/instances/{instance_id}"
    resp = requests.delete(url, headers=headers())
    if resp.status_code != 200:
        raise RuntimeError(
            f"Terminate failed ({resp.status_code}): {resp.text}"
        )
    print(f"🛑 Terminated instance {instance_id}")
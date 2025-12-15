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

def launch_instance(instance_type, region_name, ssh_key_id=None, ssh_key_name=None):
    """
    Launch a Lambda Cloud instance.
    
    Args:
        instance_type: e.g., "gpu_1x_a10", "gpu_1x_a100_80gb"
        region_name: e.g., "us-east-1", "us-west-2"
        ssh_key_id: Optional SSH key ID
        ssh_key_name: Optional SSH key name (can use either id or name)
    
    Returns:
        instance_id: The ID of the launched instance
    """
    url = f"{BASE_URL}/instance-operations/launch"

    payload = {
        "instance_type_name": instance_type,
        "region_name": region_name,
    }
    
    # Add SSH key - prefer name over id if both provided
    if ssh_key_name:
        payload["ssh_key_names"] = [ssh_key_name]
    elif ssh_key_id:
        payload["ssh_key_ids"] = [ssh_key_id]

    resp = requests.post(url, headers=headers(), json=payload)
    print("🔥 Launch response status:", resp.status_code)
    print("🔥 Launch response body:", resp.text)
    resp.raise_for_status()

    data = resp.json()
    
    # Handle different response formats
    if "data" in data:
        if "instance_ids" in data["data"] and len(data["data"]["instance_ids"]) > 0:
            instance_id = data["data"]["instance_ids"][0]
            print("🚀 Launched instance:", instance_id)
            return instance_id
        elif "operation_id" in data["data"]:
            # If we get an operation ID, poll it
            op_id = data["data"]["operation_id"]
            print(f"⏳ Got operation ID: {op_id}, polling...")
            instance_id = wait_for_operation(op_id)
            return instance_id
    
    raise RuntimeError(f"Unexpected API response format: {data}")



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


def list_instances():
    """
    List all instances in your Lambda Cloud account.
    """
    url = f"{BASE_URL}/instances"
    resp = requests.get(url, headers=headers())
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def list_ssh_keys():
    """
    List all SSH keys in your Lambda Cloud account.
    """
    url = f"{BASE_URL}/ssh-keys"
    resp = requests.get(url, headers=headers())
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def list_instance_types():
    """
    List available instance types and their availability.
    """
    url = f"{BASE_URL}/instance-types"
    resp = requests.get(url, headers=headers())
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", {})


def ssh_connect(ip: str, ssh_key_path: str = None, interactive: bool = True):
    """
    SSH into the instance interactively.
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
    ]
    print(f"🔗 Connecting to {ip}...")
    print(" ", " ".join(cmd))
    if interactive:
        subprocess.call(cmd)
    else:
        subprocess.check_call(cmd)
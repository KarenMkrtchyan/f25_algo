#!/usr/bin/env python3
"""
Sync results from Lambda Cloud instance to local machine.

Usage:
    python src/cloud/sync_results_from_lambda.py [--instance-ip IP] [--results-dir DIR]
"""

import os
import sys
import subprocess
import argparse

# Add cloud directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from lambda_launcher import list_instances, get_instance_ip

DEFAULT_SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")
REMOTE_RESULTS_DIR = "~/f25_algo/Results"
LOCAL_RESULTS_DIR = "Results"


def sync_results(instance_ip: str, remote_dir: str, local_dir: str, ssh_key_path: str):
    """Sync results directory from Lambda instance to local machine."""
    print(f"📥 Syncing results from {instance_ip}...")
    print(f"   Remote: {remote_dir}")
    print(f"   Local:  {local_dir}\n")
    
    # Expand ~ in remote path
    remote_dir_expanded = remote_dir.replace("~", f"/home/ubuntu")
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Use scp to copy files
    cmd = [
        "scp",
        "-i", ssh_key_path,
        "-r",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        f"ubuntu@{instance_ip}:{remote_dir_expanded}/*",
        local_dir + "/"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    try:
        subprocess.check_call(cmd)
        print(f"\n✅ Results synced successfully!")
        print(f"   Check: {os.path.abspath(local_dir)}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error syncing results: {e}")
        print("\n💡 Try syncing manually:")
        print(f"   scp -i {ssh_key_path} -r ubuntu@{instance_ip}:{remote_dir_expanded} {local_dir}/")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Sync results from Lambda Cloud instance to local machine"
    )
    parser.add_argument(
        "--instance-ip",
        help="IP address of Lambda instance (auto-detected if not provided)"
    )
    parser.add_argument(
        "--results-dir",
        default=REMOTE_RESULTS_DIR,
        help=f"Remote results directory (default: {REMOTE_RESULTS_DIR})"
    )
    parser.add_argument(
        "--local-dir",
        default=LOCAL_RESULTS_DIR,
        help=f"Local directory to save results (default: {LOCAL_RESULTS_DIR})"
    )
    parser.add_argument(
        "--ssh-key-path",
        default=DEFAULT_SSH_KEY_PATH,
        help=f"Path to SSH private key (default: {DEFAULT_SSH_KEY_PATH})"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("LAMBDA_API_KEY"):
        print("❌ Error: LAMBDA_API_KEY environment variable not set!")
        sys.exit(1)
    
    # Get instance IP
    if args.instance_ip:
        instance_ip = args.instance_ip
    else:
        print("📋 Finding active Lambda instance...")
        instances = list_instances()
        active_instances = [inst for inst in instances if inst.get("status") == "active"]
        
        if not active_instances:
            print("❌ No active instances found!")
            print("   Please provide --instance-ip or launch an instance first.")
            sys.exit(1)
        
        instance = active_instances[0]
        instance_ip = instance.get("ip")
        if not instance_ip:
            instance_ip = get_instance_ip(instance.get("id"))
        
        print(f"✅ Found active instance: {instance_ip}\n")
    
    # Sync results
    sync_results(
        instance_ip=instance_ip,
        remote_dir=args.results_dir,
        local_dir=args.local_dir,
        ssh_key_path=args.ssh_key_path
    )


if __name__ == "__main__":
    main()


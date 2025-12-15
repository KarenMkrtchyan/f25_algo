#!/usr/bin/env python3
"""
Simple script to connect to Lambda Cloud GPU instances.

Usage:
    python src/cloud/connect_to_lambda.py

This will:
1. List your existing instances
2. If none exist, help you launch a new one
3. Connect you via SSH to the instance
"""

import os
import sys

# Add cloud directory to path to import lambda_launcher directly
# This avoids triggering src.__init__.py which has dependencies
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from lambda_launcher import (
    list_instances,
    list_ssh_keys,
    list_instance_types,
    launch_instance,
    wait_for_instance_ready,
    get_instance_ip,
    ssh_connect,
    terminate_instance,
)

# Configuration - adjust these to your preferences
DEFAULT_INSTANCE_TYPE = "gpu_1x_a100_sxm4"  # Change to your preferred GPU type
DEFAULT_REGION = "us-east-1"  # Change to your preferred region
DEFAULT_SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")  # Path to your private SSH key


def print_instances(instances):
    """Print a formatted list of instances."""
    if not instances:
        print("📭 No instances found.")
        return
    
    print("\n🖥️  Your Lambda Cloud Instances:")
    print("=" * 60)
    for inst in instances:
        status = inst.get("status", "unknown")
        ip = inst.get("ip", "N/A")
        instance_id = inst.get("id", "N/A")
        instance_type = inst.get("instance_type", {}).get("name", "N/A")
        
        status_emoji = {
            "active": "✅",
            "booting": "⏳",
            "unhealthy": "⚠️",
            "terminated": "❌"
        }.get(status, "❓")
        
        print(f"{status_emoji} {instance_id[:8]}... | {status:12} | {instance_type:20} | {ip}")
    print("=" * 60)


def print_ssh_keys(ssh_keys):
    """Print a formatted list of SSH keys."""
    if not ssh_keys:
        print("📭 No SSH keys found.")
        return
    
    print("\n🔑 Your SSH Keys:")
    print("=" * 60)
    for key in ssh_keys:
        key_id = key.get("id", "N/A")
        key_name = key.get("name", "N/A")
        print(f"  • {key_name} (ID: {key_id[:8]}...)")
    print("=" * 60)


def print_instance_types(instance_types):
    """Print available instance types."""
    print("\n🖥️  Available Instance Types:")
    print("=" * 60)
    for name, info in instance_types.items():
        gpu = info.get("instance_type", {}).get("gpu_description", "N/A")
        regions = info.get("regions_with_capacity_available", [])
        if regions:
            print(f"  • {name:20} | {gpu:30} | Regions: {', '.join(regions[:3])}")
    print("=" * 60)


def main():
    print("🚀 Lambda Cloud Connection Helper\n")
    
    # Check API key
    if not os.getenv("LAMBDA_API_KEY"):
        print("❌ Error: LAMBDA_API_KEY environment variable not set!")
        print("\nTo set it, run:")
        print("  export LAMBDA_API_KEY='your-api-key-here'")
        print("\nOr add it to your ~/.zshrc or ~/.bashrc:")
        print("  echo 'export LAMBDA_API_KEY=\"your-api-key-here\"' >> ~/.zshrc")
        sys.exit(1)
    
    # List existing instances
    print("📋 Checking existing instances...")
    instances = list_instances()
    active_instances = [inst for inst in instances if inst.get("status") == "active"]
    
    print_instances(instances)
    
    if active_instances:
        print("\n✅ Found active instance(s)!")
        # Use the first active instance
        instance = active_instances[0]
        ip = instance.get("ip")
        instance_id = instance.get("id")
        
        if ip:
            print(f"\n🔗 Connecting to {ip} (Instance: {instance_id[:8]}...)\n")
            ssh_connect(ip, ssh_key_path=DEFAULT_SSH_KEY_PATH)
        else:
            print("⚠️  Instance is active but has no IP yet. Waiting...")
            ip = wait_for_instance_ready(instance_id)
            ssh_connect(ip, ssh_key_path=DEFAULT_SSH_KEY_PATH)
    else:
        print("\n📭 No active instances found.")
        response = input("Would you like to launch a new instance? (y/n): ").strip().lower()
        
        if response != 'y':
            print("👋 Exiting. Run this script again when you have an instance.")
            sys.exit(0)
        
        # Get SSH keys
        print("\n📋 Fetching your SSH keys...")
        ssh_keys = list_ssh_keys()
        print_ssh_keys(ssh_keys)
        
        if not ssh_keys:
            print("\n❌ No SSH keys found! Please add an SSH key in the Lambda Cloud dashboard first.")
            sys.exit(1)
        
        # Use first SSH key by default
        ssh_key = ssh_keys[0]
        ssh_key_name = ssh_key.get("name")
        ssh_key_id = ssh_key.get("id")
        
        # Show available instance types
        print("\n📋 Fetching available instance types...")
        instance_types = list_instance_types()
        print_instance_types(instance_types)
        
        # Launch instance
        print(f"\n🚀 Launching {DEFAULT_INSTANCE_TYPE} in {DEFAULT_REGION}...")
        print(f"   Using SSH key: {ssh_key_name}")
        
        try:
            instance_id = launch_instance(
                instance_type=DEFAULT_INSTANCE_TYPE,
                region_name=DEFAULT_REGION,
                ssh_key_name=ssh_key_name,
                ssh_key_id=ssh_key_id
            )
            
            print(f"\n⏳ Waiting for instance to become active...")
            ip = wait_for_instance_ready(instance_id)
            
            print(f"\n🎉 Instance is ready!")
            print(f"   Instance ID: {instance_id}")
            print(f"   IP Address: {ip}")
            
            response = input("\n🔗 Connect now? (y/n): ").strip().lower()
            if response == 'y':
                ssh_connect(ip, ssh_key_path=DEFAULT_SSH_KEY_PATH)
            else:
                print(f"\n💡 To connect later, run:")
                print(f"   ssh -i {DEFAULT_SSH_KEY_PATH} ubuntu@{ip}")
                print(f"\n⚠️  Don't forget to terminate the instance when done:")
                print(f"   python -c \"from src.cloud.lambda_launcher import terminate_instance; terminate_instance('{instance_id}')\"")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Launch interrupted. Instance may still be launching.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error launching instance: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()


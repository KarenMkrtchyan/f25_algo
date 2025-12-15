#!/usr/bin/env python3
"""
Run any Python file on Lambda Cloud GPU.

Usage:
    python src/cloud/run_file_on_lambda.py <path_to_file.py> [--instance-type TYPE] [--region REGION] [--keep-alive]

Examples:
    python src/cloud/run_file_on_lambda.py src/Experiments/Logit/check_instances.py
    python src/cloud/run_file_on_lambda.py src/Experiments/Logit/4digit_neuron.py --instance-type gpu_1x_a100_80gb
    python src/cloud/run_file_on_lambda.py my_script.py --keep-alive  # Don't terminate instance after
"""

import os
import sys
import argparse
from textwrap import dedent
from pathlib import Path

# Add cloud directory to path to import lambda_launcher directly
# This avoids triggering src.__init__.py which has dependencies
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from lambda_launcher import (
    list_instances,
    list_ssh_keys,
    launch_instance,
    wait_for_instance_ready,
    get_instance_ip,
    ssh_and_run,
    terminate_instance,
)

# Configuration
DEFAULT_INSTANCE_TYPE = "gpu_1x_a10"
DEFAULT_REGION = "us-east-1"
DEFAULT_SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")
GIT_REPO_URL = "https://github.com/KarenMkrtchyan/f25_algo.git"
REMOTE_PROJECT_DIR = "~/f25_algo"


def find_or_launch_instance(instance_type, region_name, ssh_key_name=None, ssh_key_id=None):
    """Find an active instance or launch a new one."""
    instances = list_instances()
    active_instances = [inst for inst in instances if inst.get("status") == "active"]
    
    if active_instances:
        instance = active_instances[0]
        instance_id = instance.get("id")
        ip = instance.get("ip")
        
        if not ip:
            print("⏳ Instance is active but has no IP yet. Waiting...")
            ip = wait_for_instance_ready(instance_id)
        
        print(f"✅ Using existing instance: {instance_id[:8]}... (IP: {ip})")
        return instance_id, ip
    
    # Launch new instance
    print(f"🚀 Launching new instance: {instance_type} in {region_name}...")
    
    if not ssh_key_name and not ssh_key_id:
        ssh_keys = list_ssh_keys()
        if not ssh_keys:
            raise RuntimeError("No SSH keys found! Please add one in Lambda Cloud dashboard.")
        ssh_key = ssh_keys[0]
        ssh_key_name = ssh_key.get("name")
        ssh_key_id = ssh_key.get("id")
    
    instance_id = launch_instance(
        instance_type=instance_type,
        region_name=region_name,
        ssh_key_name=ssh_key_name,
        ssh_key_id=ssh_key_id
    )
    
    print("⏳ Waiting for instance to become active...")
    ip = wait_for_instance_ready(instance_id)
    
    return instance_id, ip


def build_setup_and_run_command(script_path: str, keep_alive: bool = False) -> str:
    """
    Build a command that sets up the environment and runs the script.
    """
    # Convert script path to remote path
    # If script is in the repo, use the same relative path
    script_path = Path(script_path).as_posix()
    if script_path.startswith("src/"):
        remote_script = script_path
    else:
        # If it's outside src/, we'll copy it to a temp location
        remote_script = f"~/temp_script.py"
        # For now, assume it's in the repo structure
        remote_script = script_path
    
    cmd = dedent(f"""
    set -e
    
    echo "📦 Setting up environment..."
    
    # Update apt and install basics if needed
    if ! command -v git &> /dev/null || ! command -v python3.11 &> /dev/null; then
        echo "📦 Installing system packages..."
        sudo apt-get update -y
        sudo apt-get install -y git software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update -y
        sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils
        curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.11
    fi
    
    # Clone or update repo (expand ~ to home directory)
    REMOTE_DIR_EXPANDED=$(echo {REMOTE_PROJECT_DIR} | sed "s|~|$HOME|")
    
    if [ ! -d "$REMOTE_DIR_EXPANDED" ]; then
        echo "📥 Cloning repo..."
        git clone {GIT_REPO_URL} "$REMOTE_DIR_EXPANDED"
    elif [ -d "$REMOTE_DIR_EXPANDED/.git" ]; then
        echo "🔄 Updating repo..."
        cd "$REMOTE_DIR_EXPANDED"
        git fetch --all --quiet || true
        git reset --hard HEAD --quiet || true
        git pull --quiet || echo "⚠️  Git pull had issues, but continuing..."
    else
        echo "⚠️  Directory exists but is not a git repository"
        echo "   Removing and re-cloning..."
        rm -rf "$REMOTE_DIR_EXPANDED"
        git clone {GIT_REPO_URL} "$REMOTE_DIR_EXPANDED"
    fi
    
    cd "$REMOTE_DIR_EXPANDED"
    
    # Create venv with Python 3.11 if needed
    if [ ! -d "venv" ]; then
        echo "🐍 Creating virtualenv with Python 3.11..."
        python3.11 -m venv venv
    fi
    
    # Activate venv and install/upgrade dependencies
    source venv/bin/activate
    
    echo "📦 Installing/upgrading dependencies..."
    pip install --upgrade pip --quiet
    
    # Install PyTorch with CUDA (use version from requirements.txt if specified, otherwise default)
    # Check if requirements.txt specifies torch version
    if grep -q "^torch==" requirements.txt 2>/dev/null; then
        REQUIRED_TORCH_VERSION=$(grep "^torch==" requirements.txt | cut -d'=' -f3)
    else
        REQUIRED_TORCH_VERSION="2.5.1"
    fi
    
    # Check current torch version if installed
    CURRENT_TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
    
    if [ -z "$CURRENT_TORCH_VERSION" ] || [ "$CURRENT_TORCH_VERSION" != "$REQUIRED_TORCH_VERSION" ]; then
        echo "🔥 Installing PyTorch $REQUIRED_TORCH_VERSION with CUDA..."
        if [ -n "$CURRENT_TORCH_VERSION" ]; then
            echo "   Upgrading from torch $CURRENT_TORCH_VERSION to $REQUIRED_TORCH_VERSION"
        fi
        pip install torch==$REQUIRED_TORCH_VERSION --index-url https://download.pytorch.org/whl/cu121 --upgrade --quiet
        # Install compatible torchvision (0.20.1+ for torch 2.9.0, has InterpolationMode)
        if [ "$REQUIRED_TORCH_VERSION" = "2.9.0" ]; then
            echo "   Installing compatible torchvision 0.20.1 for torch 2.9.0..."
            pip install torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --upgrade --quiet
        else
            # For other torch versions, install latest compatible torchvision
            pip install torchvision --index-url https://download.pytorch.org/whl/cu121 --upgrade --quiet
        fi
    else
        echo "✅ PyTorch $CURRENT_TORCH_VERSION already installed (matches requirements)"
    fi
    
    # Install other dependencies from requirements.txt if it exists (excluding torch/torchvision)
    if [ -f "requirements.txt" ]; then
        echo "📦 Installing requirements (excluding torch/torchvision)..."
        # Create temp requirements file without torch/torchvision
        grep -v "^torch==" requirements.txt | grep -v "^torchvision==" > /tmp/requirements_no_torch.txt
        pip install -r /tmp/requirements_no_torch.txt --quiet
    else
        # Install common dependencies
        pip install transformer_lens einops pyarrow pandas tqdm matplotlib circuitsvis plotly --quiet
        pip install "git+https://github.com/neelnanda-io/neel-plotly.git" --quiet
    fi
    
    echo "▶️  Running script: {remote_script}"
    echo "=" * 60
    python {remote_script}
    echo "=" * 60
    echo "✅ Script finished."
    """).strip()
    
    if keep_alive:
        cmd += "\n\necho '💡 Instance kept alive. You can SSH in to continue working.'"
    
    return f"bash -lc '{cmd}'"


def main():
    parser = argparse.ArgumentParser(
        description="Run a Python file on Lambda Cloud GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
        Examples:
          %(prog)s src/Experiments/Logit/check_instances.py
          %(prog)s my_script.py --instance-type gpu_1x_a100_80gb --region us-west-2
          %(prog)s script.py --keep-alive  # Keep instance running after script completes
        """)
    )
    
    parser.add_argument(
        "script",
        help="Path to Python script to run (relative to repo root or absolute)"
    )
    parser.add_argument(
        "--instance-type",
        default=DEFAULT_INSTANCE_TYPE,
        help=f"Instance type to use (default: {DEFAULT_INSTANCE_TYPE})"
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"Region to launch in (default: {DEFAULT_REGION})"
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the instance running after the script completes"
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
        print("\nTo set it, run:")
        print("  export LAMBDA_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Check if script exists
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"❌ Error: Script not found: {args.script}")
        sys.exit(1)
    
    print(f"🚀 Running {args.script} on Lambda Cloud GPU\n")
    print(f"   Instance Type: {args.instance_type}")
    print(f"   Region: {args.region}")
    print(f"   Keep Alive: {args.keep_alive}\n")
    
    instance_id = None
    try:
        # Find or launch instance
        instance_id, ip = find_or_launch_instance(
            instance_type=args.instance_type,
            region_name=args.region
        )
        
        # Build and run the command
        remote_cmd = build_setup_and_run_command(args.script, keep_alive=args.keep_alive)
        ssh_and_run(ip, remote_cmd, ssh_key_path=args.ssh_key_path)
        
        if not args.keep_alive:
            print("\n🛑 Terminating instance...")
            terminate_instance(instance_id)
            print("✅ Done!")
        else:
            print(f"\n💡 Instance kept alive!")
            print(f"   Instance ID: {instance_id}")
            print(f"   IP: {ip}")
            print(f"   Connect with: ssh -i {args.ssh_key_path} ubuntu@{ip}")
            print(f"   Terminate with: python -c \"from src.cloud.lambda_launcher import terminate_instance; terminate_instance('{instance_id}')\"")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        if instance_id:
            response = input("Terminate instance? (y/n): ").strip().lower()
            if response == 'y':
                try:
                    terminate_instance(instance_id)
                except Exception as e:
                    print(f"⚠️  Failed to terminate: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if instance_id and not args.keep_alive:
            print("🛑 Attempting to terminate instance...")
            try:
                terminate_instance(instance_id)
            except Exception as term_e:
                print(f"⚠️  Failed to terminate: {term_e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


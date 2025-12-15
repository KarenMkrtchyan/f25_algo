#!/usr/bin/env python3
"""
Set up your Lambda Cloud instance with the repository and dependencies.

This script will:
1. Clone your repository (or update if it exists)
2. Create a virtual environment
3. Install all dependencies including PyTorch with CUDA

Usage:
    python src/cloud/setup_lambda_instance.py
"""

import os
import sys

# Add cloud directory to path to import lambda_launcher directly
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from lambda_launcher import (
    list_instances,
    wait_for_instance_ready,
    get_instance_ip,
    ssh_and_run,
)

# Configuration
DEFAULT_SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")
GIT_REPO_URL = "https://github.com/KarenMkrtchyan/f25_algo.git"
REMOTE_PROJECT_DIR = "~/f25_algo"


def build_setup_command() -> str:
    """Build the setup command to run on the remote instance."""
    from textwrap import dedent
    
    cmd = dedent(f"""
    set -e
    
    echo "🚀 Setting up Lambda Cloud instance..."
    echo "=" * 60
    
    # Update apt and install basics
    echo "📦 Installing system packages..."
    sudo apt-get update -y
    sudo apt-get install -y git software-properties-common
    
    # Install Python 3.11 from deadsnakes PPA
    echo "🐍 Installing Python 3.11..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils
    
    # Install pip for Python 3.11
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
    
    # Clone or update repo (expand ~ to home directory)
    REMOTE_DIR_EXPANDED=$(echo {REMOTE_PROJECT_DIR} | sed "s|~|$HOME|")
    
    if [ ! -d "$REMOTE_DIR_EXPANDED" ]; then
        echo "📥 Cloning repository..."
        git clone {GIT_REPO_URL} "$REMOTE_DIR_EXPANDED"
    elif [ -d "$REMOTE_DIR_EXPANDED/.git" ]; then
        echo "🔄 Repository exists, updating..."
        cd "$REMOTE_DIR_EXPANDED"
        git fetch --all --quiet || true
        git reset --hard HEAD --quiet || true
        git pull --quiet || echo "⚠️  Git pull had issues, but continuing..."
        cd ~
    else
        echo "⚠️  Directory exists but is not a git repository"
        echo "   Removing and re-cloning..."
        rm -rf "$REMOTE_DIR_EXPANDED"
        git clone {GIT_REPO_URL} "$REMOTE_DIR_EXPANDED"
    fi
    
    cd "$REMOTE_DIR_EXPANDED"
    
    # Create venv with Python 3.11 if needed
    if [ ! -d "venv" ]; then
        echo "🐍 Creating virtual environment with Python 3.11..."
        python3.11 -m venv venv
    fi
    
    # Activate venv and upgrade pip
    source venv/bin/activate
    echo "📦 Upgrading pip..."
    pip install --upgrade pip --quiet
    
    # Install PyTorch with CUDA (use version from requirements.txt if specified)
    echo "🔥 Installing PyTorch with CUDA support..."
    if grep -q "^torch==" requirements.txt 2>/dev/null; then
        REQUIRED_TORCH_VERSION=$(grep "^torch==" requirements.txt | cut -d'=' -f3)
    else
        REQUIRED_TORCH_VERSION="2.5.1"
    fi
    
    # Check current torch version if installed
    CURRENT_TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
    
    if [ -z "$CURRENT_TORCH_VERSION" ] || [ "$CURRENT_TORCH_VERSION" != "$REQUIRED_TORCH_VERSION" ]; then
        echo "   Using torch version from requirements.txt: $REQUIRED_TORCH_VERSION"
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
    
    # Install requirements (excluding torch/torchvision to avoid conflicts)
    if [ -f "requirements.txt" ]; then
        echo "📦 Installing requirements from requirements.txt (excluding torch/torchvision)..."
        grep -v "^torch==" requirements.txt | grep -v "^torchvision==" > /tmp/requirements_no_torch.txt
        pip install -r /tmp/requirements_no_torch.txt --quiet
    else
        echo "📦 Installing common dependencies..."
        pip install transformer_lens einops pyarrow pandas tqdm matplotlib circuitsvis plotly --quiet
        pip install "git+https://github.com/neelnanda-io/neel-plotly.git" --quiet
    fi
    
    echo "=" * 60
    echo "✅ Setup complete!"
    echo ""
    echo "💡 To activate the environment, run:"
    echo "   cd {REMOTE_PROJECT_DIR}"
    echo "   source venv/bin/activate"
    echo ""
    echo "💡 To run a script, use:"
    echo "   python src/Experiments/Logit/Token_prediction.py"
    """).strip()
    
    return f"bash -lc '{cmd}'"


def main():
    print("🔧 Lambda Cloud Instance Setup\n")
    
    # Check API key
    if not os.getenv("LAMBDA_API_KEY"):
        print("❌ Error: LAMBDA_API_KEY environment variable not set!")
        print("\nTo set it, run:")
        print("  export LAMBDA_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Find active instance
    print("📋 Checking for active instances...")
    instances = list_instances()
    active_instances = [inst for inst in instances if inst.get("status") == "active"]
    
    if not active_instances:
        print("❌ No active instances found!")
        print("\n💡 Launch an instance first using:")
        print("   python src/cloud/connect_to_lambda.py")
        sys.exit(1)
    
    instance = active_instances[0]
    instance_id = instance.get("id")
    ip = instance.get("ip")
    
    if not ip:
        print("⏳ Instance is active but has no IP yet. Waiting...")
        ip = wait_for_instance_ready(instance_id)
    
    print(f"✅ Found active instance: {instance_id[:8]}... (IP: {ip})")
    print(f"\n🔧 Setting up instance...\n")
    
    try:
        remote_cmd = build_setup_command()
        ssh_and_run(ip, remote_cmd, ssh_key_path=DEFAULT_SSH_KEY_PATH)
        print("\n✅ Setup complete! You can now run your scripts on the instance.")
        print(f"\n💡 To connect and run scripts:")
        print(f"   ssh -i {DEFAULT_SSH_KEY_PATH} ubuntu@{ip}")
        print(f"   cd ~/f25_algo")
        print(f"   source venv/bin/activate")
        print(f"   python src/Experiments/Logit/Token_prediction.py")
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


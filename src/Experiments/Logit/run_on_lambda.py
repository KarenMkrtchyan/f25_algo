import os
import sys
from textwrap import dedent

# Make sure we can import src.cloud
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.cloud.lambda_launcher import (
    launch_instance,
    wait_for_operation,
    get_instance_ip,
    ssh_and_run,
    terminate_instance,
    wait_for_instance_ready
)

# ==== CONFIG ====
INSTANCE_TYPE = "gpu_1x_a10"
REGION_NAME = "us-east-1"
SSH_KEY_ID = "48fc4be56983414092a63d007c50a090"  # lambda@lambda

# Path to your private key on your Mac that corresponds to that public key in Lambda
LOCAL_PRIVATE_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")  # adjust if needed

GIT_REPO_URL = GIT_REPO_URL = "https://github.com/KarenMkrtchyan/f25_algo.git"
REMOTE_PROJECT_DIR = "~/f25_algo"
REMOTE_PYTHON = "~/f25_algo/venv/bin/python"
REMOTE_SCRIPT = "src/Experiments/Logit/4digit.py"


def build_remote_setup_command() -> str:
    """
    This builds a single bash command string that will run on the remote machine.
    It:
      - installs git & python3-venv
      - clones or pulls the repo
      - creates venv if needed
      - installs basic deps
      - runs the experiment script
    You can customize this heavily later.
    """
    cmd = dedent(f"""
    set -e

    echo "📦 Updating apt & installing basics..."
    sudo apt-get update -y
    sudo apt-get install -y git python3-venv python3-pip

    if [ ! -d "{REMOTE_PROJECT_DIR}" ]; then
        echo "📥 Cloning repo..."
        git clone {GIT_REPO_URL} {REMOTE_PROJECT_DIR}
    else
        echo "🔄 Repo already there, pulling latest..."
        cd {REMOTE_PROJECT_DIR}
        git pull
    fi

    cd {REMOTE_PROJECT_DIR}

    if [ ! -d "venv" ]; then
        echo "🐍 Creating virtualenv..."
        python3 -m venv venv
    fi

    echo "📦 Installing Python dependencies..."

    # Upgrade pip first
    ~/f25_algo/venv/bin/pip install --upgrade pip

    # Install PyTorch CUDA wheels
    ~/f25_algo/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # Install safe subset of your requirements
    ~/f25_algo/venv/bin/pip install \
        transformer_lens \
        einops \
        pyarrow \
        pandas \
        tqdm \
        matplotlib \
        circuitsvis \
        plotly

    # Install neel_plotly from GitHub (not PyPI)
    ~/f25_algo/venv/bin/pip install "git+https://github.com/neelnanda-io/neel-plotly.git"

    echo "▶️ Running experiment..."
    {REMOTE_PYTHON} {REMOTE_SCRIPT}

    echo "✅ Experiment finished."
    """).strip()

    # We pass this whole thing as a single quoted string to ssh
    return f"bash -lc '{cmd}'"


def main():
    print("🚦 Launching remote activation patching workflow...")

    # 1) Launch
    instance_id = launch_instance(
    instance_type="gpu_1x_a10",
    region_name="us-east-1",
    ssh_key_id="48fc4be56983414092a63d007c50a090",
    ssh_key_name="lambda@lambda"
    )


    # 2) Wait for operation to complete and get instance id
    ip = wait_for_instance_ready(instance_id)

    try:
        # 3) Get IP & run remote setup+experiment
        ip = get_instance_ip(instance_id)
        remote_cmd = build_remote_setup_command()
        ssh_and_run(ip, remote_cmd, ssh_key_path=LOCAL_PRIVATE_KEY_PATH)
    finally:
        # 4) Terminate instance no matter what
        try:
            terminate_instance(instance_id)
        except Exception as e:
            print(f"⚠️ Failed to terminate instance automatically: {e}")
            print("Please go to Lambda Cloud console and delete it manually.")


if __name__ == "__main__":
    main()

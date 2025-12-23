#!/bin/bash
# Upgrade existing Lambda instance to use Python 3.11

cd ~/f25_algo

echo "🐍 Installing Python 3.11..."
sudo apt-get update -y
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils

# Install pip for Python 3.11
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.11

echo "🔄 Recreating virtual environment with Python 3.11..."
# Remove old venv
if [ -d "venv" ]; then
    rm -rf venv
fi

# Create new venv with Python 3.11
python3.11 -m venv venv

echo "📦 Activating new environment and installing packages..."
source venv/bin/activate

# Verify Python version
python --version

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

echo "✅ Upgrade complete! Python version:"
python --version


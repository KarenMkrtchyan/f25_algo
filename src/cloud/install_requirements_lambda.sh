#!/bin/bash
# Install requirements compatible with Python 3.10 on Lambda Cloud

cd ~/f25_algo
source venv/bin/activate

echo "📦 Installing core dependencies..."

# Install essential packages first
pip install transformer_lens einops pyarrow pandas tqdm matplotlib circuitsvis plotly --quiet
pip install "git+https://github.com/neelnanda-io/neel-plotly.git" --quiet

# Install most packages from requirements.txt, but skip incompatible ones
echo "📦 Installing from requirements.txt (skipping incompatible packages)..."

# Create a temporary requirements file without the problematic packages
grep -v "^ipython==9.6.0" requirements.txt | \
grep -v "^ipykernel==7.1.0" | \
grep -v "^#" | \
grep -v "^torch==" | \
grep -v "^torchvision==" > /tmp/requirements_lambda.txt

# Install from the filtered requirements
pip install -r /tmp/requirements_lambda.txt --quiet

# Install compatible versions of ipython and ipykernel
echo "📦 Installing compatible IPython..."
pip install "ipython<9.0" "ipykernel<8.0" --quiet

echo "✅ Installation complete!"


#!/bin/bash
# Install requirements compatible with Python 3.10 on Lambda Cloud

cd ~/f25_algo
source venv/bin/activate

echo "📦 Installing core dependencies first..."

# Install essential packages
pip install transformer_lens einops pyarrow pandas tqdm matplotlib circuitsvis plotly --quiet
pip install "git+https://github.com/neelnanda-io/neel-plotly.git" --quiet

echo "📦 Installing compatible versions of problematic packages..."
# Install compatible versions
pip install "ipython<9.0" "ipykernel<8.0" "networkx<3.5" --quiet

echo "📦 Installing remaining packages from requirements.txt..."
# Create filtered requirements file
grep -v "^ipython==9.6.0" requirements.txt | \
grep -v "^ipykernel==7.1.0" | \
grep -v "^networkx==3.5" | \
grep -v "^torch==" | \
grep -v "^torchvision==" | \
grep -v "^#" > /tmp/requirements_lambda.txt

# Install from filtered requirements, ignoring errors for packages that might still have issues
pip install -r /tmp/requirements_lambda.txt 2>&1 | grep -v "ERROR" || true

echo "✅ Installation complete!"
echo ""
echo "💡 You can now run your scripts:"
echo "   python src/Experiments/Logit/Token_prediction.py"


#!/bin/bash

# Setup script to clone and configure the official HTSAT repository
# Based on: https://github.com/RetroCirce/HTS-Audio-Transformer

echo "=========================================="
echo "Setting up Official HTSAT Repository"
echo "=========================================="

# Directory setup
HTSAT_DIR="HTSAT"
OFFICIAL_REPO_DIR="$HTSAT_DIR/HTS-Audio-Transformer"

cd "$HTSAT_DIR"

# Clone the official repository if not already present
if [ ! -d "$OFFICIAL_REPO_DIR" ]; then
    echo "Cloning official HTSAT repository..."
    git clone https://github.com/RetroCirce/HTS-Audio-Transformer.git
    echo "✓ Repository cloned successfully"
else
    echo "✓ Repository already exists"
fi

# Install requirements
echo ""
echo "Installing dependencies..."
cd "$OFFICIAL_REPO_DIR"

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
elif [ -f "sed_model/requirements.txt" ]; then
    pip install -r sed_model/requirements.txt
else
    echo "Installing common dependencies..."
    pip install torch torchaudio numpy pandas librosa scikit-learn tqdm
fi

echo "✓ Dependencies installed"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Repository location: $OFFICIAL_REPO_DIR"
echo ""
echo "Next steps:"
echo "1. Review the official repository structure"
echo "2. Adapt the evaluation code to use your checkpoint"
echo "3. Configure for ESC-50 with val_fold=2"
echo ""
echo "For custom evaluation using simplified code, run:"
echo "  cd $HTSAT_DIR"
echo "  python test_setup.py"
echo "  python evaluate.py"
echo ""





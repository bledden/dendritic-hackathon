#!/bin/bash
# AMD MI300x Quick Setup Script
# Run this on your MI300x instance to set up everything automatically

set -e  # Exit on error

echo "========================================================================"
echo "🚀 AMD MI300x Dendritic Whisper Setup"
echo "========================================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}❌ This script is for Linux only${NC}"
    exit 1
fi

# Check Python version
echo -e "\n${YELLOW}[1/8]${NC} Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}❌ Python 3.9+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"

# Check ROCm installation
echo -e "\n${YELLOW}[2/8]${NC} Checking ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    echo -e "${GREEN}✅ ROCm detected${NC}"
    rocm-smi | head -10
else
    echo -e "${YELLOW}⚠️  ROCm not detected. Please install ROCm first.${NC}"
    echo "See: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo -e "\n${YELLOW}[3/8]${NC} Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Install PyTorch for ROCm
echo -e "\n${YELLOW}[4/8]${NC} Installing PyTorch for ROCm..."
echo "This may take a few minutes..."

# Detect ROCm version
if command -v rocminfo &> /dev/null; then
    ROCM_VERSION=$(rocminfo | grep "HSA Runtime Version" | awk '{print $4}' | cut -d. -f1,2)
    echo "Detected ROCm version: $ROCM_VERSION"

    if [[ "$ROCM_VERSION" == "6.0" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
    else
        # Default to 5.7
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
    fi
else
    # Default to 5.7 if can't detect
    echo "Could not detect ROCm version, using 5.7"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
fi

echo -e "${GREEN}✅ PyTorch installed${NC}"

# Install project dependencies
echo -e "\n${YELLOW}[5/8]${NC} Installing project dependencies..."
pip install -r requirements_amd.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Install Perforated AI
echo -e "\n${YELLOW}[6/8]${NC} Installing Perforated AI..."
if [ -d "PerforatedAI" ]; then
    pip install -e ./PerforatedAI
    echo -e "${GREEN}✅ Perforated AI installed${NC}"
else
    echo -e "${RED}❌ PerforatedAI directory not found${NC}"
    echo "Please ensure PerforatedAI is in the project root"
    exit 1
fi

# Set up environment file
echo -e "\n${YELLOW}[7/8]${NC} Setting up environment..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}⚠️  Created .env from template${NC}"
        echo "Please edit .env with your API keys if needed"
    else
        echo "# Environment variables" > .env
        echo "OPENROUTER_API_KEY=your_key_here" >> .env
        echo "ANTHROPIC_API_KEY=your_key_here" >> .env
        echo -e "${YELLOW}⚠️  Created empty .env file${NC}"
        echo "Please add your API keys to .env"
    fi
else
    echo -e "${GREEN}✅ .env file exists${NC}"
fi

# Run GPU verification
echo -e "\n${YELLOW}[8/8]${NC} Verifying GPU setup..."
python test_gpu_mi300x.py

# Summary
echo ""
echo "========================================================================"
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment (if not already active):"
echo "   source venv/bin/activate"
echo ""
echo "2. Navigate to training directory:"
echo "   cd whisper-edge-optimization/dendritic"
echo ""
echo "3. Run a quick test (5 minutes):"
echo "   python train_dendritic_full.py \\"
echo "     --save-name quick_test \\"
echo "     --val-max-samples 50 \\"
echo "     --val-max-samples-per-epoch 50 \\"
echo "     --max-epochs 15 \\"
echo "     --max-dendrites 3 \\"
echo "     --batch-size 64 \\"
echo "     --device cuda"
echo ""
echo "4. Run production training (60 minutes):"
echo "   python train_dendritic_full.py \\"
echo "     --save-name production_run \\"
echo "     --val-max-samples 500 \\"
echo "     --val-max-samples-per-epoch 200 \\"
echo "     --max-epochs 50 \\"
echo "     --max-dendrites 10 \\"
echo "     --batch-size 128 \\"
echo "     --device cuda \\"
echo "     --use-wandb"
echo ""
echo "See AMD_SETUP.md for detailed documentation"
echo "========================================================================"

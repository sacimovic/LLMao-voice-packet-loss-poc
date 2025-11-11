#!/bin/bash
# Quick start script for LLMao Audio Repair ICF App

set -e

echo "🎙️  Starting LLMao Audio Repair ICF App..."
echo ""

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if dependencies are installed
if [ ! -d ".venv" ]; then
    echo "📦 Installing dependencies (this may take a few minutes)..."
    poetry install
fi

echo ""
echo "✅ Starting app..."
echo ""

poetry run python main.py

#!/bin/bash
# Quick start script for LLMao Audio Repair Copilot

set -e

cd "$(dirname "$0")"

# Check if backend URL is set
if [ -z "$BACKEND_URL" ]; then
    export BACKEND_URL="http://localhost:8000"
    echo "Using default backend URL: $BACKEND_URL"
fi

# Set log level if not already set
if [ -z "$RUST_LOG" ]; then
    export RUST_LOG="info"
fi

echo "🚀 Starting LLMao Audio Repair Copilot..."
echo "🔧 Backend URL: $BACKEND_URL"
echo "📊 Log level: $RUST_LOG"
echo ""

# Build if not already built
if [ ! -f "target/release/llmao_audio_repair_copilot" ]; then
    echo "📦 Building project..."
    cargo build --release
fi

# Run the copilot
exec cargo run --release

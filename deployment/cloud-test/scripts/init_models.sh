#!/bin/bash
# init_models.sh — Model hydration for OVMS 2025.4
# ACT-105: Downloads TinyLlama and generates model_config.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${BASE_DIR}/models"

MODEL_NAME="tinyllama"
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_FILE="${MODELS_DIR}/${MODEL_NAME}/1/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

echo "=== Model Hydration (ACT-105) ==="

# Create model directory structure (OVMS convention: model_name/version/)
mkdir -p "${MODELS_DIR}/${MODEL_NAME}/1"

# Download model if not already present
if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists: $MODEL_FILE"
else
    echo "Downloading TinyLlama GGUF..."
    curl -L -o "$MODEL_FILE" "$MODEL_URL"
    echo "Download complete: $MODEL_FILE"
fi

# Generate model_config.json
cat > "${MODELS_DIR}/model_config.json" << 'MCEOF'
{
    "model_config_list": [
        {
            "config": {
                "name": "tinyllama",
                "base_path": "/models/tinyllama",
                "target_device": "CPU",
                "nireq": 4,
                "plugin_config": {
                    "NUM_STREAMS": "2",
                    "PERFORMANCE_HINT": "LATENCY"
                }
            }
        }
    ]
}
MCEOF

echo "Model config written: ${MODELS_DIR}/model_config.json"

# Verify
if [ -f "$MODEL_FILE" ] && [ -f "${MODELS_DIR}/model_config.json" ]; then
    echo "MODEL HYDRATION PASS"
    echo "  Model: $MODEL_FILE ($(du -h "$MODEL_FILE" | cut -f1))"
    echo "  Config: ${MODELS_DIR}/model_config.json"
else
    echo "MODEL HYDRATION FAIL"
    exit 1
fi

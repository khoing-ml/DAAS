#!/bin/bash
# Setup ImageReward from source for local development and patches

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGEREWARD_LOCAL="$REPO_ROOT/third_party/ImageReward"
PY_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

echo "Setting up ImageReward from source..."
echo "Detected Python $PY_VERSION"

# Create third_party directory if it doesn't exist
mkdir -p "$REPO_ROOT/third_party"

# Clone if not already present
if [ ! -d "$IMAGEREWARD_LOCAL" ]; then
    echo "Cloning ImageReward repository..."
    git clone https://github.com/THUDM/ImageReward.git "$IMAGEREWARD_LOCAL"
else
    echo "ImageReward already cloned at $IMAGEREWARD_LOCAL"
fi

# Install compatible HF stack first to avoid tokenizers source builds on Python 3.12
echo "Installing compatible transformers/tokenizers wheels..."
pip install --upgrade "transformers==4.37.2" "tokenizers==0.15.2"

# Install in editable mode without pulling conflicting transitive dependencies
echo "Installing ImageReward in editable mode (no-deps)..."
pip install -e "$IMAGEREWARD_LOCAL" --no-deps

echo "ImageReward setup complete!"
echo "You can now edit $IMAGEREWARD_LOCAL/ImageReward/ files directly."

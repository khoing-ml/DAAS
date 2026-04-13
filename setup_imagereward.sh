#!/bin/bash
# Setup ImageReward from source for local development and patches

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGEREWARD_LOCAL="$REPO_ROOT/third_party/ImageReward"

echo "Setting up ImageReward from source..."

# Create third_party directory if it doesn't exist
mkdir -p "$REPO_ROOT/third_party"

# Clone if not already present
if [ ! -d "$IMAGEREWARD_LOCAL" ]; then
    echo "Cloning ImageReward repository..."
    git clone https://github.com/THUDM/ImageReward.git "$IMAGEREWARD_LOCAL"
else
    echo "ImageReward already cloned at $IMAGEREWARD_LOCAL"
fi

# Install in editable mode
echo "Installing ImageReward in editable mode..."
pip install -e "$IMAGEREWARD_LOCAL"

echo "ImageReward setup complete!"
echo "You can now edit $IMAGEREWARD_LOCAL/ImageReward/ files directly."

#!/usr/bin/env bash
set -euo pipefail

# Install NVM (Node Version Manager)
if [ ! -d "$HOME/.nvm" ]; then
  echo "[INFO] Installing NVM..."
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
else
  echo "[INFO] NVM already installed."
fi

# Load NVM into the current shell session
export NVM_DIR="$HOME/.nvm"
# shellcheck disable=SC1091
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install the latest LTS version of Node.js
echo "[INFO] Installing Node.js (LTS)..."
nvm install --lts

# Show versions
echo "[INFO] Installed versions:"
node -v
npm -v

echo "[INFO] Node installed! Installing node packages."

npm i

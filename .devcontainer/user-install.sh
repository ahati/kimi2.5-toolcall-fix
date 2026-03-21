#!/bin/bash
set -x

curl -fsSL https://opencode.ai/install | bash 
curl -fsSL https://claude.ai/install.sh | bash

npm install -g @openai/codex

mkdir /home/vscode/.codex

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

ln -s ${SCRIPT_DIR}/claude.settings.json /home/vscode/.claude/settings.json
ln -s ${SCRIPT_DIR}/codex.config.toml /home/vscode/.codex/config.toml

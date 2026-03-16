#!/bin/bash
set -x

curl -fsSL https://opencode.ai/install | bash 
curl -fsSL https://claude.ai/install.sh | bash

npm install -g @openai/codex

mkdir /home/vscode/.codex
ln -s /workspaces/kimi-k2.5-fix-proxy/.devcontainer/claude.settings.json /home/vscode/.claude/settings.json
ln -s /workspaces/kimi-k2.5-fix-proxy/.devcontainer/codex.config.toml /home/vscode/.codex/config.toml

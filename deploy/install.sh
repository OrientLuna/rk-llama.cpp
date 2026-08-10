#!/bin/bash
# Install the llama-server systemd unit for autostart on boot.
# NOTE: edit deploy/systemd/llama-server.service first — replace <your-user>
# and /opt/rk-llama.cpp with your actual user and checkout path.
set -e
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
echo "=== rk-llama.cpp install ==="
echo "Project dir: $PROJECT_DIR"

echo "1. Installing systemd unit..."
sudo cp deploy/systemd/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "2. Enabling autostart on boot..."
sudo systemctl enable llama-server.service

echo ""
echo "=== Done ==="
echo "Start:    sudo systemctl start llama-server"
echo "Logs:     sudo journalctl -u llama-server -f"
echo "API:      http://$(hostname -I | awk '{print $1}'):8080"
echo ""
echo "Note: load a model via the /models/load API, the web UI, or by setting"
echo "      load-on-startup = true on a section in deploy/models.ini."
echo "      First load takes ~30-60s for NPU init."

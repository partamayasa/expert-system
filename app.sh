#!/bin/bash

# ==============================================================================
# Main Configuration
# ==============================================================================
# Name of the tmux session used to run the project processes
SESSION="expert_system"

# Absolute path to the root directory of your project
ROOT_DIR="/home/expert-system"

# ==============================================================================
# Server IP Address Detection
# ==============================================================================
# Retrieve the first active local IP address of the Linux server
SERVER_IP=$(hostname -I | awk '{print $1}')

# Fallback mechanism if the server is offline or network interface is down
if [ -z "$SERVER_IP" ]; then
    SERVER_IP="127.0.0.1"
fi

# ==============================================================================
# Cleanup Process
# ==============================================================================
echo "1. Cleaning up existing ports and stale sessions..."

# Terminate legacy tmux session if currently running (redirect errors to /dev/null)
tmux kill-session -t $SESSION 2>/dev/null

# Forcefully terminate any active gunicorn processes
killall gunicorn 2>/dev/null

# Pause for 1 second to ensure system resources are fully released
sleep 1

# ==============================================================================
# Gunicorn Server Deployment
# ==============================================================================
echo "2. Launching Gunicorn Server in Window 0 (Port 5001)..."

# Initialize a new detached tmux session in the background (-d) using the $SESSION variable
tmux new-session -d -s $SESSION -n 'Gunicorn-Server'

# Dispatch commands to the 'Gunicorn-Server' window to:
# 1. Change directory to the project root
# 2. Run Gunicorn directly using the virtual environment path with 3 workers on port 5001
tmux send-keys -t $SESSION:'Gunicorn-Server' "cd ${ROOT_DIR} && ${ROOT_DIR}/.venv/bin/gunicorn --workers 3 --bind 0.0.0.0:5001 model.app:app" C-m

# Pause for 2 seconds to allow Gunicorn to bind port 5001
sleep 2

# ==============================================================================
# Deployment Summary
# ==============================================================================
echo "--------------------------------------------------------"
echo "[Status]: Expert Recommendation System started!"
echo "Gunicorn API URL: http://${SERVER_IP}:5001"
echo "To view logs, run : tmux attach -t $SESSION"
echo "--------------------------------------------------------"
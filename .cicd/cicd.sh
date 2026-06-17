#!/bin/bash

# 1. Navigate to the project directory or exit if the directory is missing
cd /home/expert-system || exit

# 2. Fetch the latest metadata from GitHub without changing local files yet
git fetch origin development

# 3. Record the current local commit ID VS the latest remote commit ID on the GitHub server
LOCAL_COMMIT=$(git rev-parse HEAD)
REMOTE_COMMIT=$(git rev-parse origin/development)

# 4. Compare commit IDs directly. If they differ, trigger the deployment process
if [ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]; then
    echo "$(date): New code detected on GitHub. Initiating deployment" >> /home/expert-system/.cicd/cicd.log

    # 5. Force the local files to match the development branch on the remote server
    git reset --hard origin/development

    # 6. Smart check on the service status before taking action using absolute paths
    if /usr/bin/systemctl is-active --quiet expert-system.service; then
        # If the service is running, trigger an instant hot-reload (Zero-Downtime, prevents 502)
        sudo /usr/bin/systemctl reload expert-system.service
        echo "$(date): expert-system.service successfully reloaded via hot-reload" >> /home/expert-system/.cicd/cicd.log
    else
        # If the service is inactive or dead, force start to bring the system back online
        sudo /usr/bin/systemctl start expert-system.service
        echo "$(date): expert-system.service was dead; forced start initiated" >> /home/expert-system/.cicd/cicd.log
    fi
fi
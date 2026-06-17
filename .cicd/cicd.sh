#!/bin/bash

# 1. Navigate to the project directory
cd /home/expert-system

# 2. Record the current local commit ID before checking for updates
OLD_COMMIT=$(git rev-parse HEAD)

# 3. Fetch the latest code from GitHub and overwrite local files
git fetch origin development
git reset --hard origin/development

# 4. Record the commit ID after the reset process
NEW_COMMIT=$(git rev-parse HEAD)

# 5. Check if the commit ID has changed (indicating new code was pulled)
if [ "$OLD_COMMIT" != "$NEW_COMMIT" ]; then
    echo "$(date): New code detected; initiating server update" >> /home/expert-system/.cicd/cicd.log

    # REQUIRED: Restart the service so Flask loads the newly updated code files
    sudo systemctl restart expert-system.service

    echo "$(date): expert-system.service successfully restarted" >> /home/expert-system/.cicd/cicd.log
fi
#!/bin/bash

# Create a datetime suffix
DATETIME_SUFFIX=$(date '+%Y%m%d_%H%M%S')

# Load env variables
set -o allexport
source .env
set +o allexport

# Ensure Redis broker is running
"$REDIS_PATH" > "$LOGS_PATH/redislog_$DATETIME_SUFFIX.txt" 2>&1 &

# Start Celery Worker for processing vercye runs
celery -A worker.celery_app worker \
  --logfile "$LOGS_PATH/celerylog_$DATETIME_SUFFIX.txt" \
  --loglevel=info \
  --concurrency=1 \
  -Q vercye_runs &

# Remove existing socket file if it exists, to avoid permissions issues
if [ -e "$SOCKET_PATH" ]; then
  rm "$SOCKET_PATH"
fi

# Start Uvicorn server running the FASTAPI app (also serving frontend and API)
uvicorn main:app --uds "$SOCKET_PATH" > "$LOGS_PATH/serverlog_$DATETIME_SUFFIX.txt" 2>&1 &

# wait for socket file to be created from uvicorn
while [ ! -e "$SOCKET_PATH" ]; do sleep 0.1; done

# Grant access to specific users only
setfacl -b "$SOCKET_PATH"
chmod 600 "$SOCKET_PATH"

IFS=',' read -ra USER_LIST <<< "$USERS"
for user in "${USER_LIST[@]}"; do
  setfacl -m u:"$user":rw "$SOCKET_PATH"
done

wait

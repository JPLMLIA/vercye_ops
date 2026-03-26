#!/usr/bin/env bash
set -Eeuo pipefail

# ============ ENV FIX (CRITICAL) ============
# Ensure we use ONLY this conda env
export PYTHONNOUSERSITE=1
export PATH="/home/sawahnr/conda-env-vercye-local/bin:$PATH"

# ============ Config & Preflight ============
DATETIME_SUFFIX="$(date '+%Y%m%d_%H%M%S')"

# Load .env if present
if [[ -f .env ]]; then
  set -o allexport
  source .env
  set +o allexport
else
  echo "[warn] .env not found; relying on environment variables."
fi

# Load NVM
export NVM_DIR="$ENV_BASE/env/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Required env vars
: "${REDIS_PATH:?Set REDIS_PATH in .env}"
: "${LOGS_PATH:=./logs}"
: "${SOCKET_PATH:=/tmp/vercye-uvicorn.sock}"
: "${USERS:=}"

mkdir -p "$LOGS_PATH"
mkdir -p static

# Cleanup
cleanup() {
  echo "[info] cleaning up background processes..."
  pkill -P $$ || true
}
trap cleanup EXIT

# ============ Build Frontend ============
echo "[info] Building frontend..."
pushd frontend >/dev/null
if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi
npm run build
rsync -a --delete dist/ ../static/
popd >/dev/null

# ============ Start Redis ============
echo "[info] Starting Redis..."
"$REDIS_PATH" > "$LOGS_PATH/redislog_${DATETIME_SUFFIX}.txt" 2>&1 &

# ============ Start Celery ============
echo "[info] Starting Celery worker..."

python -m celery -A worker.celery_app worker \
  --logfile "$LOGS_PATH/celerylog_${DATETIME_SUFFIX}-processing.txt" \
  --loglevel=info \
  --concurrency=1 \
  -Q vercye_processing &

python -m celery -A worker.celery_app worker \
  --logfile "$LOGS_PATH/celerylog_${DATETIME_SUFFIX}-prep.txt" \
  --loglevel=info \
  --concurrency=1 \
  -Q vercye_prep &

# ============ Start Uvicorn ============
echo "[info] Preparing Uvicorn socket..."
sock_dir="$(dirname "$SOCKET_PATH")"
mkdir -p "$sock_dir"
[[ -e "$SOCKET_PATH" ]] && rm -f "$SOCKET_PATH"

echo "[info] Starting Uvicorn..."
python -m uvicorn main:app --uds "$SOCKET_PATH" > "$LOGS_PATH/serverlog_${DATETIME_SUFFIX}.txt" 2>&1 &

# Wait for socket
echo -n "[info] Waiting for socket to be created"
for i in $(seq 1 1200); do
  [[ -e "$SOCKET_PATH" ]] && { echo " [ok]"; break; }
  echo -n "."
  sleep 0.1
  if [[ $i -eq 1200 ]]; then
    echo -e "\n[error] Uvicorn did not create the socket in time."
    tail -n 50 "$LOGS_PATH/serverlog_${DATETIME_SUFFIX}.txt" || true
    exit 1
  fi
done

# ============ Permissions ============
echo "[info] Setting socket permissions..."
chmod 600 "$SOCKET_PATH"

if command -v setfacl >/dev/null 2>&1; then
  setfacl -b "$SOCKET_PATH" || true
  if [[ -n "$USERS" ]]; then
    IFS=',' read -ra USER_LIST <<< "$USERS"
    for user in "${USER_LIST[@]}"; do
      user="${user//[[:space:]]/}"
      [[ -n "$user" ]] && setfacl -m u:"$user":rw "$SOCKET_PATH" || true
    done
  fi
else
  echo "[warn] setfacl not found; relying on chmod only."
fi

echo "[info] All services started. Press Ctrl+C to stop."
wait

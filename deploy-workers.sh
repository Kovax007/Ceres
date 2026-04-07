#!/bin/bash
# Deploy and restart Ceres workers on this server.
# Usage: ./deploy-workers.sh [num_gpus]

set -e
NUM_GPUS=${1:-8}
CERES_WORKER="$HOME/Ceres-worker"

echo "=== Deploy Ceres Workers ==="
echo "Server: $(hostname) ($(hostname -I | awk '{print $1}'))"

echo "1. Stopping workers..."
"$CERES_WORKER/stop_workers.sh" 2>/dev/null || true
pkill -f "Ceres.MCGS --worker" 2>/dev/null || true
sleep 2

echo "2. Pulling latest code..."
cd "$CERES_WORKER"
git pull 2>&1 | tail -3

echo "3. Building..."
dotnet build src/Ceres.MCGS/Ceres.MCGS.csproj -c Release 2>&1 | tail -3

echo "4. Starting $NUM_GPUS workers..."
"$CERES_WORKER/start_workers.sh" "$NUM_GPUS"

echo "5. Verifying..."
sleep 2
running=$(screen -ls 2>/dev/null | grep -c "worker-gpu" || true)
echo "   $running workers running"

if [ "$running" -eq "$NUM_GPUS" ]; then
    echo "=== Deploy complete ==="
else
    echo "=== WARNING: Expected $NUM_GPUS workers, found $running ==="
fi

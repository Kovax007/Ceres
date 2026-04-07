#!/bin/bash
# Stop all Ceres worker screen sessions.
# Usage: ./stop_workers.sh [num_gpus]

NUM_GPUS=${1:-8}

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  screen -S "worker-gpu${gpu}" -X quit 2>/dev/null && echo "Stopped worker-gpu${gpu}" || echo "worker-gpu${gpu} not running"
done

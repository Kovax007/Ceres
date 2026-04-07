#!/bin/bash
# Start all 8 Ceres workers (one per GPU) in screen sessions.
# Usage: ./start_workers.sh [num_gpus]
#   num_gpus: number of GPUs to start workers on (default: 8)

NUM_GPUS=${1:-8}
CERES_BIN="$HOME/Ceres-worker/artifacts/release/net10.0/Ceres.MCGS"

# Kill existing worker screens if any
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  screen -S "worker-gpu${gpu}" -X quit 2>/dev/null
done

# Start each worker in its own screen session with auto-restart loop.
# Workers self-exit (code 42) when RSS exceeds 15GB to prevent OOM.
# The loop respawns them automatically with a brief delay.
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  screen -dmS "worker-gpu${gpu}" bash -c "
    while true; do
      $CERES_BIN --worker --worker-config $HOME/Ceres-worker/worker_config_gpu${gpu}.json
      echo '[Restart] Worker GPU ${gpu} exited, restarting in 5s...'
      sleep 5
    done
  "
done

echo "Started $NUM_GPUS workers in screen sessions"
echo "List:   screen -ls | grep worker"
echo "Attach: screen -r worker-gpu0"
echo "Ports:  5100-$((5100 + NUM_GPUS - 1))"

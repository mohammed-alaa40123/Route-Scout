#!/bin/bash -e
# Wrapper to run predict_with_candidate_routes_scheduling.sh for multiple sample indices
# Usage: ./run_samples_multi.sh [start] [end]
# Defaults: start=1, end=20

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT="$SCRIPT_DIR/predict_with_candidate_routes_scheduling.sh"
START=${1:-1}
END=${2:-20}

if [ ! -x "$SCRIPT" ]; then
    # Make the script executable if it's not
    chmod +x "$SCRIPT" || true
fi

echo "Running samples from $START to $END using script: $SCRIPT"

for s in $(seq "$START" "$END"); do
    echo "\n=== SAMPLE $s ==="
    # Export SAMPLE_INDEX so the script can pick it up (script uses SAMPLE_INDEX=${SAMPLE_INDEX:-5})
    SAMPLE_INDEX=$s bash -c "mkdir -p $SCRIPT_DIR/Results && $SCRIPT" 2>&1 | tee "$SCRIPT_DIR/run_sample_${s}.log"
    echo "Saved log: $SCRIPT_DIR/run_sample_${s}.log"
done

echo "All done."
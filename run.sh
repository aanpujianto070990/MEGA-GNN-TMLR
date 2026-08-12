#!/bin/bash
# Run one or more MegaGNN configs, repeating each with a set of random seeds.
#
# Usage:
#   ./run.sh                                   # run the default config below
#   ./run.sh configs/PNA/AML-Small-HI-MEGA-PNA.yaml
#   RUNS_PER_CONFIG=5 ./run.sh configs/a.yaml configs/b.yaml

set -euo pipefail

# Number of random seeds per config (each run uses a different seed).
RUNS_PER_CONFIG="${RUNS_PER_CONFIG:-1}"

# Configs to run: any passed as arguments, otherwise the default below.
if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  CONFIGS=("configs/PNA/AML-Small-HI-MEGA-PNA.yaml")
fi

RUN_NUM=0
TOTAL_RUNS=$(( ${#CONFIGS[@]} * RUNS_PER_CONFIG ))

for CFG in "${CONFIGS[@]}"; do
  USED_SEEDS=()
  while [ "${#USED_SEEDS[@]}" -lt "$RUNS_PER_CONFIG" ]; do
    SEED=$(( (RANDOM % 100) + 1 ))
    # Skip seeds already used for this config.
    if [[ " ${USED_SEEDS[*]:-} " == *" $SEED "* ]]; then
      continue
    fi
    USED_SEEDS+=("$SEED")
    RUN_NUM=$((RUN_NUM + 1))

    echo "----------------------------------------"
    echo "Run $RUN_NUM/$TOTAL_RUNS | config: $CFG | seed: $SEED"
    python -m MegaGNN.main --cfg "$CFG" seed "$SEED"
  done
done

echo "----------------------------------------"
echo "All $TOTAL_RUNS run(s) completed."

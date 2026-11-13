#!/usr/bin/env bash
set -euo pipefail

# Run a CityLearn generative agent episode (analogous to run_sumo_sim.sh in GTA)
citylearn-simulacra \
  --preset fast_demo \
  --backend "${BACKEND:-mock}" \
  --steps "${STEPS:-24}" \
  --storage-path "${STORAGE_PATH:-storage/fast_demo}"

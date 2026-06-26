#!/usr/bin/env bash
# PyStorm one-shot installer (Linux / macOS / Git Bash).
# ====================================================
# Installs every module's dependencies, registers each module (editable) so the
# pystorm-<acro> commands and orchestrators are importable from anywhere, then
# builds the C++ kernels. A failed kernel build is non-fatal: that module runs
# on its pure-Python fallback. Requires Python >= 3.10.
#
#   ./install.sh
#
# Override the interpreter with: PYTHON=python3.11 ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"

MODULES=(
  augmented_hurricane_database
  storm_climatology_analysis
  life_cycle_simulation
  joint_distribution_model
  peaks_over_threshold
  probabilistic_simulation_technique
  reduced_storm_suite
  coastal_storm_hydrograph
)

# C++ engine build scripts (only the compute-heavy modules ship one).
BUILDS=(
  modules/peaks_over_threshold/backend/engines/cpp/build.py
  modules/probabilistic_simulation_technique/backend/engines/build.py
  modules/reduced_storm_suite/backend/engines/cpp/build.py
  modules/joint_distribution_model/backend/engines/cpp/build.py
  modules/augmented_hurricane_database/backend/engines/cpp/build.py
)

echo "[1/3] installing dependencies (requirements.txt) ..."
"$PY" -m pip install -r requirements.txt

echo "[2/3] registering modules (editable) ..."
"$PY" -m pip install -e common --no-deps
for m in "${MODULES[@]}"; do
  echo "      - $m"
  "$PY" -m pip install -e "modules/$m" --no-deps
done

echo "[3/3] building C++ kernels (optional accelerators) ..."
for b in "${BUILDS[@]}"; do
  echo "      - $b"
  "$PY" "$b" || echo "        WARN: build failed; pure-Python fallback will be used"
done

echo
echo "Done. Verify with:  $PY check_env.py"

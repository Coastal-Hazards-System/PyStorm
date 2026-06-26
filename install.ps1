# PyStorm one-shot installer (Windows PowerShell).
# ===============================================
# Installs every module's dependencies, registers each module (editable) so the
# pystorm-<acro> commands and orchestrators are importable from anywhere, then
# builds the C++ kernels. A failed kernel build is non-fatal: that module runs
# on its pure-Python fallback. Requires Python >= 3.10.
#
#   .\install.ps1
#
# Override the interpreter with:  $env:PYTHON='py -3.11'; .\install.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$py = if ($env:PYTHON) { $env:PYTHON } else { 'python' }

$modules = @(
  'augmented_hurricane_database',
  'storm_climatology_analysis',
  'life_cycle_simulation',
  'joint_distribution_model',
  'peaks_over_threshold',
  'probabilistic_simulation_technique',
  'reduced_storm_suite',
  'coastal_storm_hydrograph'
)

# C++ engine build scripts (only the compute-heavy modules ship one).
$builds = @(
  'modules/peaks_over_threshold/backend/engines/cpp/build.py',
  'modules/probabilistic_simulation_technique/backend/engines/build.py',
  'modules/reduced_storm_suite/backend/engines/cpp/build.py',
  'modules/joint_distribution_model/backend/engines/cpp/build.py',
  'modules/augmented_hurricane_database/backend/engines/cpp/build.py'
)

Write-Host '[1/3] installing dependencies (requirements.txt) ...'
& $py -m pip install -r requirements.txt

Write-Host '[2/3] registering modules (editable) ...'
& $py -m pip install -e common --no-deps
foreach ($m in $modules) {
  Write-Host "      - $m"
  & $py -m pip install -e "modules/$m" --no-deps
}

Write-Host '[3/3] building C++ kernels (optional accelerators) ...'
foreach ($b in $builds) {
  Write-Host "      - $b"
  try { & $py $b } catch { Write-Host '        WARN: build failed; pure-Python fallback will be used' }
}

Write-Host ''
Write-Host "Done. Verify with:  $py check_env.py"

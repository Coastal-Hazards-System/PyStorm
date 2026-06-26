# PyStorm task runner (Windows PowerShell). Same verbs as the Makefile.
#
#   .\tasks.ps1 install                 install everything (deps + modules + kernels)
#   .\tasks.ps1 doctor                  check the environment
#   .\tasks.ps1 build                   build all C++ kernels
#   .\tasks.ps1 test                    run every module's test suite
#   .\tasks.ps1 run rss                 run a module launcher
#   .\tasks.ps1 run rss --mode optimal --scope regional
#
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true, Position = 0)]
  [ValidateSet('install', 'doctor', 'build', 'test', 'run', 'help')]
  [string]$Command,

  [Parameter(Position = 1)]
  [string]$Module,

  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Rest
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = if ($env:PYTHON) { $env:PYTHON } else { 'python' }

$map = @{
  ahd = 'augmented_hurricane_database'; sca = 'storm_climatology_analysis'
  lcs = 'life_cycle_simulation';        jdm = 'joint_distribution_model'
  pot = 'peaks_over_threshold';         pst = 'probabilistic_simulation_technique'
  rss = 'reduced_storm_suite';          csh = 'coastal_storm_hydrograph'
}
$builds = @(
  "modules/$($map.pot)/backend/engines/cpp/build.py",
  "modules/$($map.pst)/backend/engines/build.py",
  "modules/$($map.rss)/backend/engines/cpp/build.py",
  "modules/$($map.jdm)/backend/engines/cpp/build.py",
  "modules/$($map.ahd)/backend/engines/cpp/build.py"
)

switch ($Command) {
  'help'    { Write-Host 'verbs: install | doctor | build | test | run <acro> [args]'; Write-Host 'acronyms: ahd sca lcs jdm pot pst rss csh' }
  'install' { & "$PSScriptRoot/install.ps1" }
  'doctor'  { & $py check_env.py }
  'build'   { foreach ($b in $builds) { Write-Host "build $b"; try { & $py $b } catch { Write-Host "WARN: $b failed (fallback)" } } }
  'test'    { foreach ($m in $map.Values) { Write-Host "== test $m =="; Push-Location "modules/$m"; & $py -m pytest -q; Pop-Location } }
  'run' {
    if (-not $Module -or -not $map.ContainsKey($Module)) {
      Write-Host 'usage: .\tasks.ps1 run <ahd|sca|lcs|jdm|pot|pst|rss|csh> [args]'; exit 2
    }
    $d = $map[$Module]
    & $py "modules/$d/run_$d.py" @Rest
  }
}

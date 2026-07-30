<#
.SYNOPSIS
    Runs the full tabular H1/H2 experiment sweep: every opponent arm x every seed.

.DESCRIPTION
    One set = (arms) x (seeds) training runs. Every invocation gets its OWN timestamped output
    folder (runs/<yyyy-MM-dd_HH-mm-ss>/) so a new sweep never mixes with, or is polluted by, the
    CSVs of an earlier run. Each run writes DISTINCT files via --tag inside that folder:
        runs/<stamp>/<opponent>_s<seed>_ep<episodes>.csv          final Q-table
        runs/<stamp>/<opponent>_s<seed>_ep<episodes>_best.csv      best-health-diff checkpoint
        runs/<stamp>/<opponent>_s<seed>_ep<episodes>_metrics.csv   per-eval learning curve

    After training, analyze.py is run against THIS sweep's folder only, so the aggregate/summary
    reflect the task just run (and its summary.csv / agg_*.csv / combined.csv are overwritten
    fresh inside the same folder).

    The three arms cover BOTH hypotheses in a single sweep:
        champion (= win_max)  -> H1 Win-Max arm / H2 Champion arm
        teacher  (= td_error) -> H1 TD-Error arm / H2 Teacher arm
        selfplay              -> H2 third arm (past-self self-play)

    Everything except --opponent and --seed is held identical across arms (the controls).

.EXAMPLE
    ./run_sweep.ps1
    ./run_sweep.ps1 -Episodes 50000 -Seeds 0,1,2,3,4,5,6,7 -Arms champion,teacher
#>

param(
    # Opponent arms to run. champion/teacher cover H1+H2; add/remove selfplay as needed.
    [string[]] $Arms      = @('champion', 'teacher', 'selfplay'),
    # Replication seeds (same set for every arm -> fair, averageable comparison).
    [int[]]    $Seeds     = @(0, 1, 2, 3, 4),
    # Training length per run. 20000 is comfortably past the epsilon floor for a 48x15 table.
    [int]      $Episodes  = 20000,
    [int]      $MaxSteps  = 1200,
    [ValidateSet('discrete', 'continuous')]
    [string]   $Reward    = 'discrete',
    [int]      $EvalEvery = 1000,
    [int]      $LogEvery  = 1000,
    # Parent directory for sweep outputs; each run creates a timestamped subfolder under it.
    [string]   $OutDir    = 'runs',
    # Skip the automatic analyze.py pass at the end (training CSVs are still written).
    [switch]   $SkipAnalysis,
    # Which python to call (override if it isn't on PATH as 'python').
    [string]   $Python    = 'python'
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# Isolate THIS sweep in its own timestamped folder so analyze.py only ever sees this run's CSVs
# (a flat shared 'runs/' would let stale *_metrics.csv from earlier runs pollute the aggregate).
$stamp  = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
$runDir = Join-Path $OutDir $stamp
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$total   = $Arms.Count * $Seeds.Count
$index   = 0
$started = Get-Date

Write-Host "=== Tabular sweep: $($Arms.Count) arms x $($Seeds.Count) seeds = $total runs ===" -ForegroundColor Cyan
Write-Host "    episodes=$Episodes  reward=$Reward  out=$runDir`n"

foreach ($arm in $Arms) {
    foreach ($seed in $Seeds) {
        $index++
        $tag = "${arm}_s${seed}_ep${Episodes}"
        Write-Host "[$index/$total] $tag" -ForegroundColor Yellow

        & $Python train_tabular.py `
            --opponent   $arm `
            --seed       $seed `
            --episodes   $Episodes `
            --max-steps  $MaxSteps `
            --reward     $Reward `
            --eval-every $EvalEvery `
            --log-every  $LogEvery `
            --out-dir    $runDir `
            --tag        $tag

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  run '$tag' failed (exit $LASTEXITCODE); stopping." -ForegroundColor Red
            exit $LASTEXITCODE
        }
    }
}

$elapsed = (Get-Date) - $started
Write-Host "`n=== Done: $total runs in $([math]::Round($elapsed.TotalMinutes, 1)) min ===" -ForegroundColor Green
Write-Host "Checkpoints + metrics CSVs are in: $runDir"

# Aggregate + summarize THIS run only. Output (summary.csv / agg_*.csv / combined.csv / *.png) is
# written into the same timestamped folder and overwritten fresh, so it always reflects this task.
if (-not $SkipAnalysis) {
    Write-Host "`n=== Analyzing this run ($stamp) ===" -ForegroundColor Cyan
    & $Python analyze.py --runs-dir $runDir --out-dir $runDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  analysis failed (exit $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "Aggregate + summary written to: $runDir" -ForegroundColor Green
}

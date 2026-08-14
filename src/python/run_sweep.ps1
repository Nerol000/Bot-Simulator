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

    Optional 4th arm for the sharp form of H2 (direct-improvement teacher):
        improve               -> rewards genomes by measured learner improvement, not TD error.
                                 Its signal is sparse (one number per eval), so give it a SMALL
                                 --eval-every so more blocks happen, e.g.:
        ./run_sweep.ps1 -Arms champion,teacher,improve -EvalEvery 250 -Episodes 50000

    Everything except --opponent and --seed is held identical across arms (the controls).

.EXAMPLE
    ./run_sweep.ps1
    ./run_sweep.ps1 -Episodes 50000 -Seeds 0,1,2,3,4,5,6,7 -Arms champion,teacher
    ./run_sweep.ps1 -Episodes 50000 -Arms champion,teacher,improve -EvalEvery 250
#>

param(
    # Opponent arms to run. champion/teacher cover H1+H2; add selfplay and/or improve as needed.
    # 'improve' is the H2-proper direct-improvement teacher (use a small -EvalEvery with it).
    [string[]] $Arms      = @('champion', 'teacher', 'improve'),
    # Replication seeds (same set for every arm -> fair, averageable comparison).
    [int[]]    $Seeds     = @(0, 1, 2, 3, 4, 5),
    # Training length per run. 20000 is comfortably past the epsilon floor for a 48x15 table.
    [int]      $Episodes  = 50000,
    [int]      $MaxSteps  = 1200,
    [ValidateSet('discrete', 'continuous')]
    [string]   $Reward    = 'discrete',
    [int]      $EvalEvery = 100,
    [int]      $LogEvery  = 5,
    # Parent directory for sweep outputs; each run creates a timestamped subfolder under it.
    [string]   $OutDir    = 'runs',
    # Skip the automatic analyze.py pass at the end (training CSVs are still written).
    [switch]   $SkipAnalysis,
    # Which python to call (override if it isn't on PATH as 'python').
    [string]   $Python    = 'python'
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# Resolve a python that can ACTUALLY run the trainer. Bare 'python' on Windows is often the
# Microsoft Store alias stub (in ...\WindowsApps\), which prints "Python was not found" and exits
# nonzero WITHOUT running anything. So we try the caller's -Python, then py / python3 / python,
# skip the Store stub, and pick the first that can import the trainer's one third-party dep (numpy).
# We do NOT try to rewrite the interpreter to a "real" exe path -- if a launcher mangles arguments,
# the post-run recovery step below relocates the output into the date folder regardless.
$DEP_CHECK = "import numpy"   # the one third-party import train_tabular.py needs
function Test-PythonUsable {
    param([string] $Exe)
    if (-not $Exe) { return $false }
    & $Exe -c $DEP_CHECK 2>$null
    return ($LASTEXITCODE -eq 0)
}
function Resolve-Python {
    param([string] $Preferred)
    $names = @()
    if ($Preferred) { $names += $Preferred } 
    $names += @('py', 'python3', 'python')
    foreach ($n in $names) {
        $cmd = Get-Command $n -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -like '*\WindowsApps\*') { continue }   # skip the Store alias stub
        if (Test-PythonUsable $n) { return $n }
    }
    return $null
}

$Python = Resolve-Python -Preferred $Python
if (-not $Python) {
    Write-Host "No usable Python found (tried: your -Python, py, python3, python)." -ForegroundColor Red
    Write-Host "It may be the Microsoft Store alias stub, or numpy isn't installed." -ForegroundColor Red
    Write-Host "Re-run with an explicit interpreter, e.g.:  ./run_sweep.ps1 -Python C:\path\to\python.exe" -ForegroundColor Yellow
    exit 1
}
$pyVer = (& $Python -c "import sys; print(sys.version.split()[0])" 2>$null)
Write-Host "Using Python: $Python  (v$pyVer)" -ForegroundColor DarkGray

# Reference the trainer/analyzer by ABSOLUTE path (next to this script), not by bare filename.
# Set-Location fixes cwd, but resolving explicitly means a missing file fails HERE with a clear
# message instead of python's cryptic "can't open file ... [Errno 2]". (This is exactly what bit
# the analysis step: analyze.py wasn't present in the run folder on the training machine.)
$trainScript   = Join-Path $PSScriptRoot 'train_tabular.py'
$analyzeScript = Join-Path $PSScriptRoot 'analyze.py'
if (-not (Test-Path -LiteralPath $trainScript)) {
    Write-Host "train_tabular.py not found next to this script ($trainScript)." -ForegroundColor Red
    Write-Host "Copy the whole PythonTrainer folder (train_tabular.py, analyze.py, environment.py, core/) together." -ForegroundColor Yellow
    exit 1
}

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

        # Build the arg list as an array and splat it -- more robust than backtick line-continuation
        # (no trailing-backtick pitfalls) and passes each token verbatim to the real python.exe.
        $trainArgs = @(
            $trainScript,
            '--opponent',   $arm,
            '--seed',       $seed,
            '--episodes',   $Episodes,
            '--max-steps',  $MaxSteps,
            '--reward',     $Reward,
            '--eval-every', $EvalEvery,
            '--log-every',  $LogEvery,
            '--out-dir',    $runDir,
            '--tag',        $tag
        )
        & $Python @trainArgs

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  run '$tag' failed (exit $LASTEXITCODE); stopping." -ForegroundColor Red
            exit $LASTEXITCODE
        }

        # Where the metrics file SHOULD be (inside the timestamped folder).
        $expected = Join-Path $runDir "${tag}_metrics.csv"

        # Recovery: some python launchers/shims (a `py`/`python` wrapper, conda/pyenv .bat, etc.)
        # silently DROP the --out-dir argument, so the trainer used its own default out-dir ('runs',
        # == $OutDir) and wrote the files to the PARENT instead of the timestamped subfolder. Rather
        # than depend on the launcher forwarding args correctly, just relocate any stray files for
        # THIS tag from $OutDir into $runDir. (Get-ChildItem -File does not recurse, so files already
        # correctly inside runs/<stamp>/ are never touched.)
        if (-not (Test-Path $expected)) {
            $strays = Get-ChildItem -Path $OutDir -Filter "${tag}*.csv" -File -ErrorAction SilentlyContinue
            if ($strays) {
                foreach ($f in $strays) { Move-Item -LiteralPath $f.FullName -Destination $runDir -Force }
                Write-Host "  (relocated $($strays.Count) stray file(s) from '$OutDir' into the run folder -- launcher dropped --out-dir)" -ForegroundColor DarkYellow
            }
        }

        # After recovery, the metrics CSV must exist; if not, the run genuinely produced nothing.
        if (-not (Test-Path $expected)) {
            Write-Host "  run '$tag' exited 0 but wrote no metrics ($expected missing)." -ForegroundColor Red
            Write-Host "  reproduce: $Python $($trainArgs -join ' ')" -ForegroundColor Yellow
            exit 1
        }
    }
}

$elapsed = (Get-Date) - $started
Write-Host "`n=== Done: $total runs in $([math]::Round($elapsed.TotalMinutes, 1)) min ===" -ForegroundColor Green
Write-Host "Checkpoints + metrics CSVs are in: $runDir"

# Aggregate + summarize THIS run only. Output (summary.csv / agg_*.csv / combined.csv / *.png) is
# written into the same timestamped folder and overwritten fresh, so it always reflects this task.
if (-not $SkipAnalysis) {
    # Guard: if training produced no metrics (e.g. every run crashed), the folder is empty and
    # analyze.py would fail confusingly. Surface the real problem instead.
    $metrics = Get-ChildItem -Path $runDir -Filter '*_metrics.csv' -ErrorAction SilentlyContinue
    if (-not $metrics) {
        Write-Host "`nNo *_metrics.csv were written to $runDir -- training produced no output; skipping analysis." -ForegroundColor Red
        exit 1
    }
    Write-Host "`n=== Analyzing this run ($stamp) ===" -ForegroundColor Cyan
    if (-not (Test-Path -LiteralPath $analyzeScript)) {
        Write-Host "analyze.py not found next to this script ($analyzeScript); skipping analysis." -ForegroundColor Red
        Write-Host "Copy analyze.py into the same folder as run_sweep.ps1, then run:" -ForegroundColor Yellow
        Write-Host "    $Python analyze.py --runs-dir `"$runDir`" --out-dir `"$runDir`"" -ForegroundColor Yellow
        exit 1
    }
    & $Python @($analyzeScript, '--runs-dir', $runDir, '--out-dir', $runDir)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  analysis failed (exit $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "Aggregate + summary written to: $runDir" -ForegroundColor Green
}
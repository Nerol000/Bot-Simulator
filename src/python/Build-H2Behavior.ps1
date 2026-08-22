# Builds the H2-A "behavioral differences" table from the per-run <arm>_s<seed>_ep<N>_behavior.csv
# files that train_tabular.py writes. For each run it takes the CONVERGED behavior (mean over the
# last -LastN eval rows), averages across seeds per arm, and emits a Metric x Arm table matching the
# paper's H2 table (Attack % / Retreat % / Strafe % / Avg Distance, plus approach/idle for context).
#
# Usage:
#   .\Build-H2Behavior.ps1 -ResultsDir runs\<timestamp>
#   .\Build-H2Behavior.ps1 -ResultsDir runs\<timestamp> -LastN 10
param(
    [string]$ResultsDir = (Join-Path $PSScriptRoot '..\2026-07-31_13-12-49'),
    # how many trailing eval rows to average as the "converged" behavior of a run
    [int]$LastN = 5
)

# code arm tag -> paper label (H2-A compares champion vs teacher; improve shown if present)
$ArmLabel = [ordered]@{
    champion = 'Champion FSM'; teacher = 'TD-Max FSM'; td_error = 'TD-Max FSM';
    improve = 'Improvement FSM'; win_max = 'Win-Max FSM'; selfplay = 'Self-Play FSM'
}
$ArmOrder = @('champion', 'win_max', 'teacher', 'td_error', 'improve', 'selfplay')

# behavior columns -> friendly metric row labels (rates rendered as %, distance as blocks)
$MetricRows = [ordered]@{
    attack_rate = 'Attack %'; retreat_rate = 'Retreat %'; strafe_rate = 'Strafe %';
    approach_rate = 'Approach %'; idle_rate = 'Idle %'; avg_distance = 'Avg Distance'
}

$files = Get-ChildItem -Path $ResultsDir -Filter '*_behavior.csv' -File -ErrorAction SilentlyContinue
if (-not $files) { throw "No *_behavior.csv in '$ResultsDir'. Run the sweep with the behavior-logging trainer first." }

$re = '^(?<arm>.+)_s(?<seed>\d+)_ep(?<ep>\d+)_behavior\.csv$'
# perArm[arm][metric] = list of per-seed converged values
$perArm = @{}
foreach ($f in $files) {
    if ($f.Name -notmatch $re) { Write-Output "  (skip, unrecognized name) $($f.Name)"; continue }
    $arm = $Matches.arm
    $rows = @(Import-Csv $f.FullName)
    if ($rows.Count -eq 0) { Write-Output "  (skip, empty) $($f.Name)"; continue }
    $tail = $rows | Select-Object -Last $LastN
    if (-not $perArm.ContainsKey($arm)) {
        $perArm[$arm] = @{}
        foreach ($m in $MetricRows.Keys) { $perArm[$arm][$m] = @() }
    }
    foreach ($m in $MetricRows.Keys) {
        $vals = @($tail | ForEach-Object { [double]$_.$m })
        if ($vals.Count -gt 0) {
            $mean = ($vals | Measure-Object -Average).Average
            $perArm[$arm][$m] += , $mean   # one converged value for this seed
        }
    }
}

# arms present, ordered (known first, then any extras)
$armsPresent = @($ArmOrder | Where-Object { $perArm.ContainsKey($_) }) +
               @($perArm.Keys | Where-Object { $ArmOrder -notcontains $_ } | Sort-Object)
$armsPresent = $armsPresent | Select-Object -Unique
Write-Output ("Arms found: " + ($armsPresent -join ', '))

function Format-Cell([double[]]$vals, [string]$metric) {
    if (-not $vals -or $vals.Count -eq 0) { return '' }
    $mean = ($vals | Measure-Object -Average).Average
    $sd = if ($vals.Count -ge 2) {
        $m = $mean; [math]::Sqrt((($vals | ForEach-Object { ($_ - $m) * ($_ - $m) } | Measure-Object -Sum).Sum) / ($vals.Count - 1))
    }
    else { 0.0 }
    if ($metric -eq 'avg_distance') {
        return ('{0:N2} +/- {1:N2}' -f $mean, $sd)
    }
    # rates are fractions -> percent
    return ('{0:N1}% +/- {1:N1}' -f ($mean * 100), ($sd * 100))
}

# Build the table: one row per metric, one column per arm.
$table = foreach ($mKey in $MetricRows.Keys) {
    $row = [ordered]@{ Metric = $MetricRows[$mKey] }
    foreach ($arm in $armsPresent) {
        $label = if ($ArmLabel.Contains($arm)) { $ArmLabel[$arm] } else { $arm }
        $row[$label] = Format-Cell $perArm[$arm][$mKey] $mKey
    }
    [pscustomobject]$row
}

$out = Join-Path $ResultsDir 'h2_behavior_table.csv'
$table | Export-Csv -NoTypeInformation $out
$firstMetric = @($MetricRows.Keys)[0]
$nSeeds = @($perArm[$armsPresent[0]][$firstMetric]).Count
Write-Output "Wrote $out  (converged = mean of last $LastN evals; $nSeeds seeds per arm)."
Write-Output ''
$table | Format-Table -Auto

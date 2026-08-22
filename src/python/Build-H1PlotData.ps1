# Builds h1_plot_ready.csv: HD-EMA mean and +/-1 standard-error band bounds per arm,
# joined on episode, ready to import into Excel/Sheets to draw shaded variance bands.
param(
    [string]$ResultsDir = (Join-Path $PSScriptRoot '..\2026-07-31_13-12-49'),
    [ValidateSet('se', 'std', 'ci95')][string]$Band = 'se'
)

Set-Location $ResultsDir

# two-sided 95% t-multipliers by degrees of freedom (df = n_seeds - 1); 1.96 for large samples.
$T95 = @{ 1 = 12.706; 2 = 4.303; 3 = 3.182; 4 = 2.776; 5 = 2.571; 6 = 2.447; 7 = 2.365;
    8 = 2.306; 9 = 2.262; 10 = 2.228; 11 = 2.201; 12 = 2.179; 13 = 2.160; 14 = 2.145
}
function Get-T95([int]$df) { if ($T95.ContainsKey($df)) { $T95[$df] } else { 1.96 } }

# Band half-width from the per-episode std across seeds, using that row's actual seed count.
function Get-Half([double]$std, [int]$n) {
    if ($n -le 1) { return 0.0 }
    $se = $std / [math]::Sqrt($n)
    switch ($Band) {
        'std' { return $std }
        'se' { return $se }
        'ci95' { return (Get-T95 ($n - 1)) * $se }
    }
}

function Load($f) {
    Import-Csv $f | ForEach-Object {
        $n = [int]$_.n_seeds
        [pscustomobject]@{ ep = [int]$_.episode; m = [double]$_.hdiff_ema_mean; h = Get-Half ([double]$_.hdiff_ema_std) $n }
    }
}

# code arm tag -> friendly column prefix used in the output CSV header (paper opponent names).
$ArmLabel = @{
    champion = 'Champion'; win_max = 'WinMax'; teacher = 'TDMax'; td_error = 'TDMax';
    selfplay = 'SelfPlay'; improve = 'Improvement'
}
# preferred column order; unknown arms are appended alphabetically so nothing is dropped.
$ArmOrder = @('champion', 'win_max', 'teacher', 'td_error', 'selfplay', 'improve')

# Discover whichever agg_<arm>.csv this sweep produced instead of assuming a fixed three.
$aggFiles = Get-ChildItem -Path $ResultsDir -Filter 'agg_*.csv' -File -ErrorAction SilentlyContinue
if (-not $aggFiles) { throw "No agg_*.csv in '$ResultsDir'. Run analyze.py first." }
$armData = @{}
$armsPresent = @()
$seenLabel = @{}
foreach ($f in $aggFiles) {
    $arm = $f.BaseName -replace '^agg_', ''
    $label = if ($ArmLabel.ContainsKey($arm)) { $ArmLabel[$arm] } else { $arm }
    if ($seenLabel.ContainsKey($label)) {
        Write-Output "  (skip duplicate '$arm' -> '$label' already loaded)"
        continue
    }
    $seenLabel[$label] = $true
    $map = @{}; Load $f.FullName | ForEach-Object { $map[$_.ep] = $_ }
    $armData[$arm] = @{ label = $label; map = $map }
    $armsPresent += $arm
}
# order: known arms first (ARM_ORDER), then any extras alphabetically
$ordered = @($ArmOrder | Where-Object { $armsPresent -contains $_ }) +
           @($armsPresent | Where-Object { $ArmOrder -notcontains $_ } | Sort-Object)
Write-Output ("Arms found: " + ($ordered -join ', '))

# episode axis = union across all arms (arms should share the same eval schedule)
$eps = ($armData.Values.map.Keys | Sort-Object -Unique)
$rows = foreach ($e in $eps) {
    $row = [ordered]@{ episode = $e }
    foreach ($arm in $ordered) {
        $lbl = $armData[$arm].label
        $r = $armData[$arm].map[$e]
        if ($null -ne $r) {
            $row["${lbl}_mean"] = [math]::Round($r.m, 4)
            $row["${lbl}_lo"] = [math]::Round($r.m - $r.h, 4)
            $row["${lbl}_hi"] = [math]::Round($r.m + $r.h, 4)
        }
        else {
            $row["${lbl}_mean"] = ''; $row["${lbl}_lo"] = ''; $row["${lbl}_hi"] = ''
        }
    }
    [pscustomobject]$row
}

$out = Join-Path $ResultsDir "h1_plot_ready_$Band.csv"
$rows | Export-Csv -NoTypeInformation $out
Write-Output "Wrote $out ($($rows.Count) rows, band=$Band)."
Write-Output 'First 3 rows:'
$rows | Select-Object -First 3 | Format-Table -Auto
Write-Output 'Last 3 rows (final HD-EMA):'
$rows | Select-Object -Last 3 | Format-Table -Auto

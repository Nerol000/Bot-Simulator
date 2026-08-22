# Builds h1_plot_ready.csv: HD-EMA mean and +/-1 standard-error band bounds per arm,
# joined on episode, ready to import into Excel/Sheets to draw shaded variance bands.
param(
    [string]$ResultsDir = (Join-Path $PSScriptRoot '..\2026-07-31_13-12-49'),
    [ValidateSet('se', 'std', 'ci95')][string]$Band = 'se'
)

Set-Location $ResultsDir
$n = 5
$sqrtN = [math]::Sqrt($n)
# two-sided 95% t for df=4
$t95 = 2.776

function Get-Half([double]$std) {
    switch ($Band) {
        'std' { return $std }
        'se' { return $std / $sqrtN }
        'ci95' { return $t95 * $std / $sqrtN }
    }
}

function Load($f) {
    Import-Csv $f | ForEach-Object {
        [pscustomobject]@{ ep = [int]$_.episode; m = [double]$_.hdiff_ema_mean; h = Get-Half([double]$_.hdiff_ema_std) }
    }
}

$champ = @{}; Load 'agg_champion.csv' | ForEach-Object { $champ[$_.ep] = $_ }
$win = @{}; Load 'agg_selfplay.csv'  | ForEach-Object { $win[$_.ep] = $_ }
$td = @{}; Load 'agg_teacher.csv'   | ForEach-Object { $td[$_.ep] = $_ }

$eps = $champ.Keys | Sort-Object
$rows = foreach ($e in $eps) {
    [pscustomobject]@{
        episode       = $e
        Champion_mean = [math]::Round($champ[$e].m, 4)
        Champion_lo   = [math]::Round($champ[$e].m - $champ[$e].h, 4)
        Champion_hi   = [math]::Round($champ[$e].m + $champ[$e].h, 4)
        WinMax_mean   = [math]::Round($win[$e].m, 4)
        WinMax_lo     = [math]::Round($win[$e].m - $win[$e].h, 4)
        WinMax_hi     = [math]::Round($win[$e].m + $win[$e].h, 4)
        TDMax_mean    = [math]::Round($td[$e].m, 4)
        TDMax_lo      = [math]::Round($td[$e].m - $td[$e].h, 4)
        TDMax_hi      = [math]::Round($td[$e].m + $td[$e].h, 4)
    }
}

$out = "h1_plot_ready_$Band.csv"
$rows | Export-Csv -NoTypeInformation $out
Write-Output "Wrote $out ($($rows.Count) rows, band=$Band)."
Write-Output 'First 3 rows:'
$rows | Select-Object -First 3 | Format-Table -Auto
Write-Output 'Last 3 rows (final HD-EMA):'
$rows | Select-Object -Last 3 | Format-Table -Auto

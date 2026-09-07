<#
.SYNOPSIS
    H3 significance test: compares learners trained against different opponents to see whether
    they develop measurably different combat behaviors.

.DESCRIPTION
    Reads the per-run learner-behavior files (<arm>_s<seed>_ep<N>_learner_behavior.csv) produced
    by run_sweep.ps1 -LogLearnerBehavior, takes each run's CONVERGED behavior (mean of the last
    -LastN eval rows), then for every behavior metric runs Welch's two-sample, two-tailed t-test
    between each pair of arms. Prints mean +/- sd per arm and the t / df / p for each comparison.

    This does NOT touch H1 or H2 -- it only reads *_learner_behavior.csv, which exist only when a
    sweep was run with -LogLearnerBehavior.

.EXAMPLE
    ./Test-H3Behavior.ps1 -ResultsDir runs_all\2026-09-06_12-00-00
    ./Test-H3Behavior.ps1 -ResultsDir runs_h3\<stamp> -LastN 10
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$ResultsDir,
    # trailing eval rows to average as each run's converged behavior
    [int]$LastN = 5
)

$ErrorActionPreference = 'Stop'

$ArmLabel = [ordered]@{
    champion = 'Champion'; win_max = 'Win-Max'; teacher = 'TD-Max'; td_error = 'TD-Max';
    improve = 'Improvement'; selfplay = 'Self-Play'
}
$ArmOrder = @('champion', 'win_max', 'teacher', 'td_error', 'improve', 'selfplay')
$Metrics = @('attack_rate', 'retreat_rate', 'strafe_rate', 'approach_rate', 'idle_rate', 'avg_distance')

$re = '^(?<arm>.+)_s(?<seed>\d+)_ep(?<ep>\d+)_learner_behavior\.csv$'
$files = Get-ChildItem -Path $ResultsDir -Filter '*_learner_behavior.csv' -File -ErrorAction SilentlyContinue
if (-not $files) { throw "No *_learner_behavior.csv in '$ResultsDir'. Run the sweep with -LogLearnerBehavior first." }

# perArm[arm][metric] = list of per-seed converged values
$perArm = @{}
foreach ($f in $files) {
    if ($f.Name -notmatch $re) { continue }
    $arm = $Matches.arm
    $rows = @(Import-Csv $f.FullName)
    if ($rows.Count -eq 0) { continue }
    $tail = $rows | Select-Object -Last $LastN
    if (-not $perArm.ContainsKey($arm)) {
        $perArm[$arm] = @{}
        foreach ($m in $Metrics) { $perArm[$arm][$m] = New-Object System.Collections.Generic.List[double] }
    }
    foreach ($m in $Metrics) {
        $vals = @($tail | ForEach-Object { [double]$_.$m })
        if ($vals.Count -gt 0) {
            $mean = ($vals | Measure-Object -Average).Average
            $perArm[$arm][$m].Add($mean)
        }
    }
}

$arms = @($ArmOrder | Where-Object { $perArm.ContainsKey($_) }) +
        @($perArm.Keys | Where-Object { $ArmOrder -notcontains $_ } | Sort-Object)
$arms = $arms | Select-Object -Unique
Write-Output ("Arms found: " + (($arms | ForEach-Object { if ($ArmLabel.Contains($_)) { $ArmLabel[$_] } else { $_ } }) -join ', '))
Write-Output ("Seeds per arm: " + (@($arms | ForEach-Object { $perArm[$_][$Metrics[0]].Count }) -join ', '))
Write-Output ''

function Stat($vals) {
    $n = $vals.Count
    $m = ($vals | Measure-Object -Average).Average
    $ss = 0.0; foreach ($x in $vals) { $ss += ($x - $m) * ($x - $m) }
    $var = if ($n -gt 1) { $ss / ($n - 1) } else { 0.0 }
    [pscustomobject]@{ n = $n; mean = $m; sd = [math]::Sqrt($var); var = $var }
}

# --- Student t two-tailed p via regularized incomplete beta (Numerical Recipes betai) ---
function LogGamma($xx) {
    $cof = 76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5
    $x = $xx; $y = $xx; $tmp = $x + 5.5; $tmp -= ($x + 0.5) * [math]::Log($tmp)
    $ser = 1.000000000190015
    for ($j = 0; $j -le 5; $j++) { $y += 1; $ser += $cof[$j] / $y }
    return -$tmp + [math]::Log(2.5066282746310005 * $ser / $x)
}
function BetaCF($a, $b, $x) {
    $MAXIT = 200; $EPS = 3.0e-12; $FPMIN = 1.0e-300
    $qab = $a + $b; $qap = $a + 1.0; $qam = $a - 1.0
    $c = 1.0; $d = 1.0 - $qab * $x / $qap
    if ([math]::Abs($d) -lt $FPMIN) { $d = $FPMIN }
    $d = 1.0 / $d; $h = $d
    for ($m = 1; $m -le $MAXIT; $m++) {
        $m2 = 2 * $m
        $aa = $m * ($b - $m) * $x / (($qam + $m2) * ($a + $m2))
        $d = 1.0 + $aa * $d; if ([math]::Abs($d) -lt $FPMIN) { $d = $FPMIN }
        $c = 1.0 + $aa / $c; if ([math]::Abs($c) -lt $FPMIN) { $c = $FPMIN }
        $d = 1.0 / $d; $h *= $d * $c
        $aa = -($a + $m) * ($qab + $m) * $x / (($a + $m2) * ($qap + $m2))
        $d = 1.0 + $aa * $d; if ([math]::Abs($d) -lt $FPMIN) { $d = $FPMIN }
        $c = 1.0 + $aa / $c; if ([math]::Abs($c) -lt $FPMIN) { $c = $FPMIN }
        $d = 1.0 / $d; $del = $d * $c; $h *= $del
        if ([math]::Abs($del - 1.0) -lt $EPS) { break }
    }
    return $h
}
function BetaI($a, $b, $x) {
    if ($x -le 0.0) { return 0.0 }
    if ($x -ge 1.0) { return 1.0 }
    $bt = [math]::Exp((LogGamma ($a + $b)) - (LogGamma $a) - (LogGamma $b) +
          $a * [math]::Log($x) + $b * [math]::Log(1.0 - $x))
    if ($x -lt ($a + 1.0) / ($a + $b + 2.0)) { return $bt * (BetaCF $a $b $x) / $a }
    else { return 1.0 - $bt * (BetaCF $b $a (1.0 - $x)) / $b }
}
function TwoTailedP($t, $df) {
    if ($df -le 0) { return 1.0 }
    return (BetaI ($df / 2.0) 0.5 ($df / ($df + $t * $t)))
}

foreach ($metric in $Metrics) {
    Write-Output "=== $metric ==="
    # per-arm mean +/- sd
    foreach ($arm in $arms) {
        $s = Stat $perArm[$arm][$metric]
        $lbl = if ($ArmLabel.Contains($arm)) { $ArmLabel[$arm] } else { $arm }
        if ($metric -eq 'avg_distance') {
            "  {0,-12} {1,7:N2} +/- {2:N2}" -f $lbl, $s.mean, $s.sd
        } else {
            "  {0,-12} {1,6:N1}% +/- {2:N1}" -f $lbl, ($s.mean * 100), ($s.sd * 100)
        }
    }
    # pairwise Welch t-tests
    for ($i = 0; $i -lt $arms.Count; $i++) {
        for ($j = $i + 1; $j -lt $arms.Count; $j++) {
            $a = Stat $perArm[$arms[$i]][$metric]
            $b = Stat $perArm[$arms[$j]][$metric]
            $sea = $a.var / [math]::Max($a.n, 1)
            $seb = $b.var / [math]::Max($b.n, 1)
            $se = [math]::Sqrt($sea + $seb)
            if ($se -eq 0) { $t = 0.0; $df = 1.0 }
            else {
                $t = ($b.mean - $a.mean) / $se
                $df = ($sea + $seb) * ($sea + $seb) /
                      (($sea * $sea) / [math]::Max($a.n - 1, 1) + ($seb * $seb) / [math]::Max($b.n - 1, 1))
            }
            $p = TwoTailedP $t $df
            $sig = if ($p -lt 0.05) { '*' } else { '' }
            $la = if ($ArmLabel.Contains($arms[$i])) { $ArmLabel[$arms[$i]] } else { $arms[$i] }
            $lb = if ($ArmLabel.Contains($arms[$j])) { $ArmLabel[$arms[$j]] } else { $arms[$j] }
            "    {0,-12} vs {1,-12}  t={2,6:N2}  df={3,5:N1}  p={4,7:N4} {5}" -f $la, $lb, $t, $df, $p, $sig
        }
    }
    Write-Output ''
}
Write-Output "(* = p < 0.05, two-tailed. With multiple metrics/pairs, consider a Bonferroni correction.)"

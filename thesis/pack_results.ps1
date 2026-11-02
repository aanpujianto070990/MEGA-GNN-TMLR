<#
Packs the main-experiment results kept outside Kaggle (one folder per session,
hasil_kaggle\results_main_sesi*\results_main\...) into one zip for
thesis/analysis.py. Upload the zip to Kaggle as a private dataset and run
Analisis_Tahap6_Kaggle.ipynb with it as Input.

  powershell -ExecutionPolicy Bypass -File thesis\pack_results.ps1 `
      -Root "C:\...\project_aml\hasil_kaggle" -Out "C:\...\project_aml\hasil_tesis_bundle.zip" [-IncludeVal]

Zip layout:
  bundle_manifest.csv                     one line per packed run
  runs/<session>/<run folder>/<seed>/     run_summary.json, val/stats.json, scores_test.npz
                                          (+ scores_val.npz with -IncludeVal)
  meta/meta_Small-HI.csv.gz, meta/meta_Small-LI.csv.gz   (thesis/metadata.py)

Scores are packed only for runs analysis.py treats as valid: baselines and
temporal runs with the delta_t horizon (temporal.horizon_days > 0). Superseded
runs keep run_summary.json and val/stats.json (preliminary analysis).
Smoke tests and MEGA-GIN runs are left out.
#>
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$MetaDir = "",
    [switch]$IncludeVal
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$runRe = '^AML-Small-(HI|LI)-Multi-GIN-(base|full|t2v|sin|decay)-gpu\d+$'
$summaries = Get-ChildItem -Path (Join-Path $Root 'results_main_*\results_main\AML-*\*\run_summary.json') |
    Sort-Object FullName

if (-not $MetaDir) {
    $meta = Get-ChildItem -Path $Root -Recurse -Filter 'meta_Small-*.csv.gz' | Sort-Object FullName
} else {
    $meta = Get-ChildItem -Path $MetaDir -Filter 'meta_Small-*.csv.gz'
}
foreach ($ds in 'HI', 'LI') {
    if (-not ($meta | Where-Object { $_.Name -eq "meta_Small-$ds.csv.gz" })) {
        throw "meta_Small-$ds.csv.gz not found (pass -MetaDir)"
    }
}

if (Test-Path -LiteralPath $Out) { Remove-Item -LiteralPath $Out }
$zip = [System.IO.Compression.ZipFile]::Open($Out, 'Create')
$store = [System.IO.Compression.CompressionLevel]::NoCompression   # npz / gz are compressed already
$deflate = [System.IO.Compression.CompressionLevel]::Optimal
$seen = @{}
$manifest = New-Object System.Collections.Generic.List[string]
$manifest.Add('session,run,seed,valid,scores')
$nValid = 0; $nOther = 0
try {
    foreach ($s in $summaries) {
        $seedDir = $s.Directory
        $runDir = $seedDir.Parent
        $session = $runDir.Parent.Parent.Name
        if ($runDir.Name -notmatch $runRe) { continue }
        $cfg = $Matches[2]
        $summary = Get-Content -LiteralPath $s.FullName -Raw | ConvertFrom-Json
        $horizon = 0
        if ($summary.temporal -and $summary.temporal.horizon_days) { $horizon = [double]$summary.temporal.horizon_days }
        $valid = ($cfg -eq 'base') -or ($horizon -gt 0)
        # Same variants as run_variant() in analysis.py, so a superseded run does not hide another one.
        $variant = 'valid'
        if (-not $valid) {
            $variant = 'no_horizon'
            if ($summary.temporal.encoding -ne 'none' -and $summary.temporal.proj_init -ne 'zero') { $variant = 'old_init' }
        }
        $key = "$($runDir.Name)/$($seedDir.Name)/$variant"
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = $true

        $prefix = "runs/$session/$($runDir.Name)/$($seedDir.Name)"
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $s.FullName, "$prefix/run_summary.json", $deflate)
        $stats = Join-Path $seedDir.FullName 'val\stats.json'
        if (Test-Path -LiteralPath $stats) {
            [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $stats, "$prefix/val/stats.json", $deflate)
        }
        $scores = 'no'
        if ($valid) {
            $files = @('scores_test.npz')
            if ($IncludeVal) { $files += 'scores_val.npz' }
            foreach ($name in $files) {
                $path = Join-Path $seedDir.FullName $name
                if (Test-Path -LiteralPath $path) {
                    [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $path, "$prefix/$name", $store)
                    $scores = 'yes'
                } else {
                    Write-Warning "$prefix has no $name"
                }
            }
            $nValid++
        } else {
            $nOther++
        }
        $manifest.Add("$session,$($runDir.Name),$($seedDir.Name),$valid,$scores")
    }
    foreach ($ds in 'HI', 'LI') {
        $m = $meta | Where-Object { $_.Name -eq "meta_Small-$ds.csv.gz" } | Select-Object -First 1
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $m.FullName, "meta/$($m.Name)", $store)
    }
    $entry = $zip.CreateEntry('bundle_manifest.csv')
    $w = New-Object System.IO.StreamWriter($entry.Open(), (New-Object System.Text.UTF8Encoding($false)))
    try { foreach ($line in $manifest) { $w.WriteLine($line) } } finally { $w.Dispose() }
} finally {
    $zip.Dispose()
}
$size = (Get-Item -LiteralPath $Out).Length / 1MB
"valid runs: $nValid | superseded runs (summary only): $nOther | zip: {0:N0} MB -> $Out" -f $size

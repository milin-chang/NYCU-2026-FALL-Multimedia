param(
    [int]$Count = 4000,
    [int]$Seed = 42,
    [int]$Parallel = 12
)

$ErrorActionPreference = 'Stop'
if ($Count -lt 1 -or $Count -gt 60000) { throw 'Count must be between 1 and 60000.' }
if ($Parallel -lt 1) { throw 'Parallel must be positive.' }

$root = Split-Path -Parent $PSScriptRoot
$dataset = Join-Path $root 'dataset'
$images = Join-Path $dataset 'ffhq_1024'
New-Item -ItemType Directory -Path $images -Force | Out-Null
$manifest = Join-Path $dataset ("ffhq_train_seed{0}_{1}_ids.txt" -f $Seed, $Count)

if (Test-Path -LiteralPath $manifest) {
    $ids = @(Get-Content -LiteralPath $manifest | ForEach-Object { [int]$_ })
    if ($ids.Count -ne $Count -or @($ids | Sort-Object -Unique).Count -ne $Count -or @($ids | Where-Object { $_ -lt 0 -or $_ -ge 60000 }).Count) {
        throw "Invalid existing manifest: $manifest"
    }
} else {
    # Fisher-Yates sampling without replacement from the official FFHQ train IDs.
    [int[]]$pool = 0..59999
    $rng = [System.Random]::new($Seed)
    for ($i = $pool.Length - 1; $i -gt 0; $i--) {
        $j = $rng.Next($i + 1)
        $tmp = $pool[$i]; $pool[$i] = $pool[$j]; $pool[$j] = $tmp
    }
    $ids = @($pool[0..($Count - 1)] | Sort-Object)
    [System.IO.File]::WriteAllLines($manifest, [string[]]@($ids | ForEach-Object { '{0:D5}' -f $_ }))
}

function Test-FFHQ-Png([string]$path) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
    $stream = [System.IO.File]::OpenRead($path)
    try {
        if ($stream.Length -lt 100000) { return $false }
        $header = New-Object byte[] 24
        if ($stream.Read($header, 0, 24) -ne 24) { return $false }
        $signature = [byte[]](137,80,78,71,13,10,26,10)
        for ($i = 0; $i -lt 8; $i++) { if ($header[$i] -ne $signature[$i]) { return $false } }
        if ($header[12] -ne 73 -or $header[13] -ne 72 -or $header[14] -ne 68 -or $header[15] -ne 82) { return $false }
        $width = [System.Net.IPAddress]::NetworkToHostOrder([BitConverter]::ToInt32([byte[]]$header[16..19], 0))
        $height = [System.Net.IPAddress]::NetworkToHostOrder([BitConverter]::ToInt32([byte[]]$header[20..23], 0))
        if ($width -ne 1024 -or $height -ne 1024) { return $false }
        $stream.Seek(-12, [System.IO.SeekOrigin]::End) | Out-Null
        $footer = New-Object byte[] 12
        if ($stream.Read($footer, 0, 12) -ne 12) { return $false }
        $ending = [byte[]](0,0,0,0,73,69,78,68,174,66,96,130)
        for ($i = 0; $i -lt 12; $i++) { if ($footer[$i] -ne $ending[$i]) { return $false } }
        return $true
    } finally { $stream.Dispose() }
}

$config = Join-Path $dataset '_ffhq_curl.cfg'
for ($round = 1; $round -le 4; $round++) {
    $missing = [System.Collections.Generic.List[int]]::new()
    foreach ($id in $ids) {
        $name = '{0:D5}.png' -f $id
        $path = Join-Path $images $name
        if (-not (Test-FFHQ-Png $path)) {
            if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
            $missing.Add($id)
        }
    }
    Write-Host ("Round {0}: {1}/{2} valid images" -f $round, ($Count - $missing.Count), $Count)
    if ($missing.Count -eq 0) { break }

    $lines = [System.Collections.Generic.List[string]]::new()
    foreach ($id in $missing) {
        $name = '{0:D5}.png' -f $id
        $part = [int][math]::Floor($id / 10000) + 1
        $url = "https://huggingface.co/datasets/marcosv/ffhq-dataset/resolve/main/Part$part/$name`?download=true"
        $path = (Join-Path $images $name).Replace('\', '/')
        $lines.Add("url = `"$url`"")
        $lines.Add("output = `"$path`"")
        $lines.Add('location')
        $lines.Add('fail')
        $lines.Add('retry = 3')
        $lines.Add('retry-delay = 2')
        $lines.Add('connect-timeout = 20')
        $lines.Add('max-time = 180')
        $lines.Add('next')
    }
    $lines.RemoveAt($lines.Count - 1)
    [System.IO.File]::WriteAllLines($config, $lines)
    & curl.exe --parallel --parallel-max $Parallel --silent --show-error --config $config
    Write-Host ("curl exit code: {0}" -f $LASTEXITCODE)
}

$valid = 0
$bytes = [long]0
foreach ($id in $ids) {
    $path = Join-Path $images ('{0:D5}.png' -f $id)
    if (Test-FFHQ-Png $path) {
        $valid++
        $bytes += (Get-Item -LiteralPath $path).Length
    }
}
Write-Host ("Final: {0}/{1} valid PNGs, {2:N2} GiB" -f $valid, $Count, ($bytes / 1GB))
if ($valid -ne $Count) { throw 'Some images are missing or invalid. Re-run this script to resume.' }
if (Test-Path -LiteralPath $config) { Remove-Item -LiteralPath $config -Force }

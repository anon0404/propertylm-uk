<#
.SYNOPSIS
    Batch-matches UK property addresses against a running PropertyLM UK API.

.DESCRIPTION
    Reads addresses from an input CSV (a column named "address"), calls the
    /match endpoint for each row, and writes matched UPRNs/status/confidence
    to an output CSV. No Python required on the host - only the running
    Docker stack (./scripts/start_runtime.sh) and PowerShell.

.PARAMETER InputCsv
    Path to a CSV file with an "address" column, one address per row.

.PARAMETER OutputCsv
    Path to write results to. Defaults to <InputCsv>.matched.csv

.PARAMETER ApiUrl
    Base URL of the running API. Defaults to http://127.0.0.1:8000

.PARAMETER TopK
    Number of candidates to request per address. Defaults to 5.

.EXAMPLE
    ./scripts/batch_match.ps1 -InputCsv .\addresses.csv

.EXAMPLE
    ./scripts/batch_match.ps1 -InputCsv .\addresses.csv -OutputCsv .\out.csv -TopK 3
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$InputCsv,

    [string]$OutputCsv,

    [string]$ApiUrl = "http://127.0.0.1:8000",

    [int]$TopK = 5
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $InputCsv)) {
    throw "Input CSV not found: $InputCsv"
}

if (-not $OutputCsv) {
    $OutputCsv = [System.IO.Path]::ChangeExtension($InputCsv, $null).TrimEnd('.') + ".matched.csv"
}

try {
    Invoke-RestMethod -Uri "$ApiUrl/health" -Method Get | Out-Null
} catch {
    throw "PropertyLM API is not reachable at $ApiUrl/health. Run .\scripts\start_runtime.sh (via Git Bash or WSL) first."
}

$rows = Import-Csv -Path $InputCsv
if (-not ($rows | Get-Member -Name "address" -MemberType NoteProperty)) {
    throw "Input CSV must have an 'address' column."
}

$results = New-Object System.Collections.Generic.List[object]
$total = $rows.Count
$i = 0

foreach ($row in $rows) {
    $i++
    $address = $row.address
    $started = Get-Date

    $record = [ordered]@{
        address              = $address
        predicted_uprn       = ""
        predicted_status     = ""
        predicted_confidence = ""
        retrieval_mode       = ""
        latency_ms           = 0
        error                = ""
    }

    try {
        $body = @{ address = $address; top_k = $TopK } | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "$ApiUrl/match" -Method Post -ContentType "application/json" -Body $body
        $best = $response.best

        $record.predicted_uprn       = $best.property.uprn
        $record.predicted_status     = $best.status
        $record.predicted_confidence = $best.confidence
        $record.retrieval_mode       = $response.retrieval_mode
    } catch {
        $record.error = $_.Exception.Message
    }

    $record.latency_ms = [math]::Round(((Get-Date) - $started).TotalMilliseconds, 2)
    $results.Add([PSCustomObject]$record)

    if ($i % 25 -eq 0 -or $i -eq $total) {
        Write-Host "Matched $i/$total"
    }
}

$results | Export-Csv -Path $OutputCsv -NoTypeInformation
Write-Host ""
Write-Host "Wrote $OutputCsv"

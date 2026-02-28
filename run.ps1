param(
    [string]$File = "data/train.csv",
    [string]$TestFile = "data/test.csv",
    [string]$SampleSubmission = "data/sample_submission.csv",
    [string]$CompetitionDescription = "data/data_description.txt",
    [string]$Target = "",
    [string]$SubmissionFile = "submissions.csv",
    [string]$MetricObjective = "auto",
    [int]$MaxIterations = 2,
    [switch]$ShowLangGraph,
    [switch]$ForceTune
)

$ErrorActionPreference = "Stop"

# Support GNU-style args passed via run.cmd, e.g. --max-iterations 1 --show-langgraph
if ($args.Count -gt 0) {
    for ($i = 0; $i -lt $args.Count; $i++) {
        $token = [string]$args[$i]
        switch ($token) {
            "--file" {
                if ($i + 1 -lt $args.Count) { $File = [string]$args[++$i] }
            }
            "--test-file" {
                if ($i + 1 -lt $args.Count) { $TestFile = [string]$args[++$i] }
            }
            "--target" {
                if ($i + 1 -lt $args.Count) { $Target = [string]$args[++$i] }
            }
            "--sample-submission" {
                if ($i + 1 -lt $args.Count) { $SampleSubmission = [string]$args[++$i] }
            }
            "--competition-description" {
                if ($i + 1 -lt $args.Count) { $CompetitionDescription = [string]$args[++$i] }
            }
            "--submission-file" {
                if ($i + 1 -lt $args.Count) { $SubmissionFile = [string]$args[++$i] }
            }
            "--metric-objective" {
                if ($i + 1 -lt $args.Count) { $MetricObjective = [string]$args[++$i] }
            }
            "--max-iterations" {
                if ($i + 1 -lt $args.Count) { $MaxIterations = [int]$args[++$i] }
            }
            "--show-langgraph" { $ShowLangGraph = $true }
            "--force-tune" { $ForceTune = $true }
        }
    }
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = Join-Path $root "..\.venv\Scripts\python.exe"
}
if (-not (Test-Path $python)) {
    $python = "python"
}

if ([string]::IsNullOrWhiteSpace($Target)) {
    if (Test-Path $SampleSubmission) {
        $sampleHeader = Get-Content $SampleSubmission -TotalCount 1
        if (-not [string]::IsNullOrWhiteSpace($sampleHeader)) {
            $sampleColumns = $sampleHeader.Split(',') | ForEach-Object { $_.Trim() }
            if ($sampleColumns.Count -ge 2) {
                $nonIdColumns = @($sampleColumns | Where-Object { $_ -and $_.ToLower() -ne "id" })
                if ($nonIdColumns.Count -gt 0) {
                    $Target = $nonIdColumns[-1]
                } else {
                    $Target = $sampleColumns[-1]
                }
                Write-Host "[run.ps1] Auto-detected target from sample submission: $Target"
            }
            elseif ($sampleColumns.Count -eq 1 -and $sampleColumns[0] -ne "id") {
                $Target = $sampleColumns[0]
                Write-Host "[run.ps1] Auto-detected target from sample submission: $Target"
            }
        }
    }
}

if ([string]::IsNullOrWhiteSpace($Target)) {
    if (-not (Test-Path $File)) {
        throw "Training file not found: $File"
    }

    $header = Get-Content $File -TotalCount 1
    if ([string]::IsNullOrWhiteSpace($header)) {
        throw "Unable to detect target from empty header in $File"
    }

    $columns = $header.Split(',') | ForEach-Object { $_.Trim() }

    $preferredTargets = @("Calories", "diagnosed_diabetes", "target", "label", "class")
    $matched = $preferredTargets | Where-Object { $columns -contains $_ } | Select-Object -First 1

    if ($matched) {
        $Target = $matched
    }
    else {
        $Target = $columns[-1]
    }

    Write-Host "[run.ps1] Auto-detected target: $Target"
}

$args = @(
    "main.py",
    "--file", $File,
    "--test-file", $TestFile,
    "--sample-submission", $SampleSubmission,
    "--competition-description", $CompetitionDescription,
    "--target", $Target,
    "--submission-file", $SubmissionFile,
    "--metric-objective", $MetricObjective,
    "--max-iterations", $MaxIterations
)

if ($ShowLangGraph) {
    $args += "--show-langgraph"
}
if ($ForceTune) {
    $args += "--force-tune"
}

& $python @args

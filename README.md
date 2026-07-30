# PropertyLM UK

Local UK property-address matching using embeddings, deterministic address
identity checks, cross-encoder reranking and Qdrant.

## Runtime requirements

- Docker Desktop or Docker Engine
- Git
- A compatible `uk_properties` Qdrant snapshot supplied separately

Python is not required on the host for runtime use.

## Start locally (macOS / Linux)

```bash
git clone https://github.com/anon0404/propertylm-uk.git
cd propertylm-uk
cp .env.example .env
./scripts/start_runtime.sh
```

Then download and restore the `uk_properties` snapshot as described below, and
run `./scripts/smoke_test.sh` to confirm the matcher works.

## Windows setup

### Prerequisites

- **Docker Desktop** with the WSL2 backend enabled ([docs](https://docs.docker.com/desktop/install/windows-install/))
- **Git for Windows** ([git-scm.com](https://git-scm.com/download/win)) — this also installs Git Bash, which is used below to run the repo's `.sh` scripts
- **GitHub CLI** (`gh`) — recommended for downloading release assets ([cli.github.com](https://cli.github.com/))

All commands below are PowerShell unless marked "Git Bash".

### 1. Clone

```powershell
git clone https://github.com/anon0404/propertylm-uk.git
cd propertylm-uk
Copy-Item .env.example .env
```

### 2. Install and start the runtime

Open **Git Bash** in the repo folder and run:

```bash
./scripts/start_runtime.sh
```

This builds the API image and starts Qdrant + the API via `docker compose`.
Leave it running — the rest of the steps use PowerShell in a separate window.

### 3. Assemble the Qdrant snapshot from the GitHub release

The [v1.0.0 release](https://github.com/anon0404/propertylm-uk/releases/tag/v1.0.0)
splits the `uk_properties` snapshot into 21 parts so each asset stays under
GitHub's per-file upload limit. Download and reassemble them:

```powershell
mkdir snapshot_download
cd snapshot_download

# Downloads every asset attached to the release (~28 GB total)
gh release download v1.0.0 --repo anon0404/propertylm-uk --pattern "uk_properties-v1.snapshot.*" --pattern "chunks.sha256"
```

Verify every part before assembling:

```powershell
$ok = $true
Get-Content chunks.sha256 | ForEach-Object {
    $expected, $file = $_ -split "\s+", 2
    $actual = (Get-FileHash -Algorithm SHA256 $file.Trim()).Hash.ToLower()
    if ($actual -ne $expected) {
        Write-Host "MISMATCH: $file" -ForegroundColor Red
        $ok = $false
    }
}
if ($ok) { Write-Host "All parts verified." -ForegroundColor Green }
```

Concatenate the parts in order into a single snapshot file. Use `cmd /c copy /b`
rather than PowerShell's `Get-Content`, which is not binary-safe for a file
this size:

```powershell
cmd /c copy /b uk_properties-v1.snapshot.part-000+uk_properties-v1.snapshot.part-001+uk_properties-v1.snapshot.part-002+uk_properties-v1.snapshot.part-003+uk_properties-v1.snapshot.part-004+uk_properties-v1.snapshot.part-005+uk_properties-v1.snapshot.part-006+uk_properties-v1.snapshot.part-007+uk_properties-v1.snapshot.part-008+uk_properties-v1.snapshot.part-009+uk_properties-v1.snapshot.part-010+uk_properties-v1.snapshot.part-011+uk_properties-v1.snapshot.part-012+uk_properties-v1.snapshot.part-013+uk_properties-v1.snapshot.part-014+uk_properties-v1.snapshot.part-015+uk_properties-v1.snapshot.part-016+uk_properties-v1.snapshot.part-017+uk_properties-v1.snapshot.part-018+uk_properties-v1.snapshot.part-019+uk_properties-v1.snapshot.part-020 uk_properties-v1.snapshot
```

Then verify the assembled file against the full-file checksum:

```powershell
$expected = (Get-Content uk_properties-v1.snapshot.sha256) -split "\s+" | Select-Object -First 1
$actual = (Get-FileHash -Algorithm SHA256 uk_properties-v1.snapshot).Hash.ToLower()
if ($actual -eq $expected) { Write-Host "Snapshot OK" -ForegroundColor Green } else { Write-Host "Snapshot CORRUPT - re-download" -ForegroundColor Red }
```

### 4. Restore the snapshot into Qdrant

Back in **Git Bash**, from the repo root:

```bash
./scripts/restore_snapshot.sh /path/to/snapshot_download/uk_properties-v1.snapshot
```

(Convert the Windows path to a Git Bash path, e.g.
`/c/Users/you/propertylm-uk/snapshot_download/uk_properties-v1.snapshot`.)
Restoring ~18M points can take several minutes.

### 5. Verify it's working

```bash
./scripts/smoke_test.sh
```

or from PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Uri http://127.0.0.1:8000/match -Method Post -ContentType "application/json" -Body '{"address":"80 latimer road exeter ex4 7jp","top_k":5}'
```

## Batch address matching

To match many addresses at once against the running API, use
[`scripts/batch_match.ps1`](scripts/batch_match.ps1). It requires no Python on
the host and works anywhere PowerShell does — Windows PowerShell, or
PowerShell 7 (`pwsh`) on macOS/Linux/WSL.

Prepare a CSV with a single `address` column:

```
address
"80 latimer road exeter ex4 7jp"
"16 Alesmore WS13 8FD"
```

Then run (Windows PowerShell):

```powershell
./scripts/batch_match.ps1 -InputCsv .\addresses.csv -OutputCsv .\matched.csv
```

or with PowerShell 7 on macOS/Linux/WSL:

```bash
pwsh ./scripts/batch_match.ps1 -InputCsv addresses.csv -OutputCsv matched.csv
```

This calls `/match` once per row and writes `predicted_uprn`,
`predicted_status`, `predicted_confidence`, `retrieval_mode`, and per-row
latency to `matched.csv`. Use `-TopK` to change how many candidates are
requested per address, and `-ApiUrl` if the API is reachable somewhere other
than `http://127.0.0.1:8000`.

If you're scoring matches against known-correct UPRNs (e.g. for benchmarking
rather than plain batch matching), see
[`scripts/evaluate_addressbase_benchmark.py`](scripts/evaluate_addressbase_benchmark.py)
instead — it expects `query` and `expected_uprn` columns and reports
top-1/top-k accuracy.

# Download official Tesseract fast language data into this project only.
param([string]$Revision = 'main')
$ErrorActionPreference = 'Stop'
$OcrDirectory = Join-Path (Split-Path $PSScriptRoot -Parent) 'data/tessdata'
New-Item -ItemType Directory -Path $OcrDirectory -Force | Out-Null
$OcrFiles = @()
foreach ($Language in @('chi_sim', 'eng')) {
    $OcrTarget = Join-Path $OcrDirectory "$Language.traineddata"
    $OcrUrl = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/$Revision/$Language.traineddata"
    if (-not (Test-Path -LiteralPath $OcrTarget)) {
        Invoke-WebRequest -Uri $OcrUrl -OutFile "$OcrTarget.partial" -TimeoutSec 60
        Move-Item -LiteralPath "$OcrTarget.partial" -Destination $OcrTarget
    }
    $OcrFiles += @{ language = $Language; source = $OcrUrl; sha256 = (Get-FileHash -LiteralPath $OcrTarget -Algorithm SHA256).Hash.ToLower() }
}
ConvertTo-Json -InputObject $OcrFiles -Depth 3 | Set-Content -LiteralPath (Join-Path $OcrDirectory 'manifest.json') -Encoding utf8
Write-Output "OCR language data ready: $OcrDirectory"

# SPDX-License-Identifier: AGPL-3.0-only
# Captures every THME Studio tab as a PNG for the README.
#
# Builds thme-gui, runs it in capture mode (THME_GUI_CAPTURE_DIR), and converts the
# BMP frames it writes into PNGs under assets/screenshots/. The window appears on
# screen for a few seconds while the tabs are captured.
#
# Usage (from the repository root):  powershell -File scripts/capture_studio.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "assets/screenshots"
$raw = Join-Path $root "target/studio-capture"

cargo build --release --bin thme-gui
if (Test-Path $raw) { Remove-Item -Recurse -Force $raw }
New-Item -ItemType Directory -Force $raw, $out | Out-Null

$env:THME_GUI_CAPTURE_DIR = $raw
try {
    $proc = Start-Process -FilePath (Join-Path $root "target/release/thme-gui.exe") -PassThru
    if (-not $proc.WaitForExit(120000)) { $proc.Kill(); throw "thme-gui did not finish capturing within 120 s" }
} finally {
    Remove-Item Env:THME_GUI_CAPTURE_DIR
}

Add-Type -AssemblyName System.Drawing
$frames = Get-ChildItem $raw -Filter *.bmp | Sort-Object Name
if ($frames.Count -eq 0) { throw "No frames were captured" }
foreach ($frame in $frames) {
    $image = [System.Drawing.Image]::FromFile($frame.FullName)
    try {
        $png = Join-Path $out ($frame.BaseName + ".png")
        $image.Save($png, [System.Drawing.Imaging.ImageFormat]::Png)
        "{0}  {1}x{2}" -f (Split-Path $png -Leaf), $image.Width, $image.Height
    } finally {
        $image.Dispose()
    }
}

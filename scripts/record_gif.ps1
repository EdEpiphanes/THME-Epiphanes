# SPDX-License-Identifier: AGPL-3.0-only
# Records README GIFs from the real THME Studio (no mock-ups).
#
# Builds thme-gui, runs it in recording mode (THME_GUI_CAPTURE_DIR + THME_GUI_RECORD) for each
# clip below, and encodes the recorded frames with scripts/make_gif.py (needs Python + Pillow).
# The window appears on screen while each clip is recorded.
#
# Usage (from the repository root):  powershell -File scripts/record_gif.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "target/release/thme-gui.exe"
$raw = Join-Path $root "target/studio-record"
$out = Join-Path $root "assets/gifs"

# name, tab index, frames, crop (x0,y0,x1,y1 at the default 1380x880 window), extra env
$clips = @(
    @{ Name = "live-3d-manifold"; Tab = 10; Frames = 72; Crop = "0,320,1062,842"; Mode = "manifold" },
    @{ Name = "live-3d-slabs"; Tab = 10; Frames = 60; Crop = "0,320,1062,842"; Mode = "slabs" },
    @{ Name = "dashboard"; Tab = 0; Frames = 45; Crop = "0,0,1380,880"; Mode = "manifold" }
)

cargo build --release --bin thme-gui
# A real python first; on Windows `python` may be only the Microsoft Store stub, then use `py`.
$cmd = Get-Command python -ErrorAction SilentlyContinue
$python = if ($cmd -and $cmd.Source -notlike "*\WindowsApps\*") { "python" } else { "py" }
New-Item -ItemType Directory -Force $out | Out-Null

foreach ($clip in $clips) {
    if (Test-Path $raw) { Remove-Item -Recurse -Force $raw }
    New-Item -ItemType Directory -Force $raw | Out-Null
    $env:THME_GUI_CAPTURE_DIR = $raw
    $env:THME_GUI_RECORD = "{0}:{1}" -f $clip.Tab, $clip.Frames
    $env:THME_GUI_LIVE_MODE = $clip.Mode
    try {
        $proc = Start-Process -FilePath $exe -PassThru
        if (-not $proc.WaitForExit(180000)) { $proc.Kill(); throw "thme-gui did not finish recording $($clip.Name)" }
    } finally {
        Remove-Item Env:THME_GUI_CAPTURE_DIR, Env:THME_GUI_RECORD, Env:THME_GUI_LIVE_MODE
    }
    & $python (Join-Path $root "scripts/make_gif.py") --frames $raw --out (Join-Path $out "$($clip.Name).gif") --crop $clip.Crop --width 760 --fps 15 --colors 24
}

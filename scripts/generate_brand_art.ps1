# Builds THME's manga brand art from the mascot (assets/brand/thme-mascot.png):
#   assets/brand/thme-icon-256.png  ink-only face crop for the app icon and favicon
#   assets/brand/thme-icon-64.png
#   assets/brand/thme-banner.png    manga-panel title banner for the README
# Usage (repository root): powershell -File scripts/generate_brand_art.ps1

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
$root = Split-Path -Parent $PSScriptRoot
$brand = Join-Path $root "assets/brand"
$mascot = [System.Drawing.Bitmap]::FromFile((Join-Path $brand "thme-mascot.png"))

function Save-Png($bitmap, $name) {
    $path = Join-Path $brand $name
    $bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    "{0}  {1}x{2}  {3:N0} KB" -f $name, $bitmap.Width, $bitmap.Height, ((Get-Item $path).Length / 1KB)
}

function New-Graphics($bitmap) {
    $g = [System.Drawing.Graphics]::FromImage($bitmap)
    $g.SmoothingMode = "AntiAlias"
    $g.InterpolationMode = "HighQualityBicubic"
    $g.TextRenderingHint = "AntiAliasGridFit"
    $g
}

# --- Icon: face crop, thresholded to pure ink on paper, inside a black panel frame ---------
function New-Icon($size) {
    $icon = New-Object System.Drawing.Bitmap $size, $size
    $g = New-Graphics $icon
    $g.Clear([System.Drawing.Color]::White)
    $src = New-Object System.Drawing.Rectangle 330, 40, 560, 560
    $frame = [int][Math]::Max(2, $size / 32)
    $dst = New-Object System.Drawing.Rectangle $frame, $frame, ($size - 2 * $frame), ($size - 2 * $frame)
    $g.DrawImage($mascot, $dst, $src, [System.Drawing.GraphicsUnit]::Pixel)
    $g.Dispose()
    for ($y = 0; $y -lt $size; $y++) {
        for ($x = 0; $x -lt $size; $x++) {
            $c = $icon.GetPixel($x, $y)
            $v = if (($c.R + $c.G + $c.B) / 3 -lt 150) { 11 } else { 255 }
            $icon.SetPixel($x, $y, [System.Drawing.Color]::FromArgb(255, $v, $v, $v))
        }
    }
    $g = New-Graphics $icon
    $pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(255, 11, 11, 11)), $frame
    $g.DrawRectangle($pen, $frame / 2, $frame / 2, $size - $frame, $size - $frame)
    $g.Dispose()
    $icon
}

$icon256 = New-Icon 256
Save-Png $icon256 "thme-icon-256.png"
$icon64 = New-Icon 64
Save-Png $icon64 "thme-icon-64.png"

# --- Banner: manga title panel ---------------------------------------------------------------
$w, $h = 1280, 480
$banner = New-Object System.Drawing.Bitmap $w, $h
$g = New-Graphics $banner
$g.Clear([System.Drawing.Color]::White)
$ink = [System.Drawing.Color]::FromArgb(255, 11, 11, 11)
$inkBrush = New-Object System.Drawing.SolidBrush $ink

# Speed lines radiating from behind the title.
$focus = New-Object System.Drawing.PointF 360, 230
$rand = New-Object System.Random 7
for ($i = 0; $i -lt 150; $i++) {
    $angle = $rand.NextDouble() * 2 * [Math]::PI
    $r0 = 230 + $rand.NextDouble() * 70
    $r1 = 900
    $pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(255, 11, 11, 11)), (0.6 + $rand.NextDouble() * 1.8)
    $g.DrawLine($pen, [float]($focus.X + $r0 * [Math]::Cos($angle)), [float]($focus.Y + $r0 * [Math]::Sin($angle)), [float]($focus.X + $r1 * [Math]::Cos($angle)), [float]($focus.Y + $r1 * [Math]::Sin($angle)))
}
# White field behind the title so the lettering stays clean.
$g.FillEllipse([System.Drawing.Brushes]::White, 10, 10, 790, 460)

# Screentone: halftone dots fading toward the mascot.
for ($y = 18; $y -lt $h; $y += 12) {
    for ($x = 650; $x -lt 860; $x += 12) {
        $r = 3.6 * (($x - 650) / 210.0)
        if ($r -gt 0.4) { $g.FillEllipse($inkBrush, [float]($x - $r), [float]($y - $r), [float](2 * $r), [float](2 * $r)) }
    }
}

# Mascot panel on the right, slanted manga-style.
$panel = New-Object System.Drawing.Drawing2D.GraphicsPath
$panel.AddPolygon(@(
    (New-Object System.Drawing.PointF 790, 0), (New-Object System.Drawing.PointF $w, 0),
    (New-Object System.Drawing.PointF $w, $h), (New-Object System.Drawing.PointF 730, $h)))
$g.SetClip($panel)
$g.DrawImage($mascot, (New-Object System.Drawing.Rectangle 700, -40, 600, 600), (New-Object System.Drawing.Rectangle 0, 0, 1024, 1024), [System.Drawing.GraphicsUnit]::Pixel)
$g.ResetClip()
$g.DrawPath((New-Object System.Drawing.Pen $ink, 6), $panel)

# Lettering.
$family = if ((New-Object System.Drawing.Text.InstalledFontCollection).Families.Name -contains "Arial Black") { "Arial Black" } else { "Arial" }
$title = New-Object System.Drawing.Font $family, 150, ([System.Drawing.FontStyle]::Bold), ([System.Drawing.GraphicsUnit]::Pixel)
$sub = New-Object System.Drawing.Font "Segoe UI", 22, ([System.Drawing.FontStyle]::Bold), ([System.Drawing.GraphicsUnit]::Pixel)
$small = New-Object System.Drawing.Font "Segoe UI", 20, ([System.Drawing.FontStyle]::Regular), ([System.Drawing.GraphicsUnit]::Pixel)
$path = New-Object System.Drawing.Drawing2D.GraphicsPath
$path.AddString("THME", $title.FontFamily, [int][System.Drawing.FontStyle]::Bold, 150, (New-Object System.Drawing.PointF 96, 92), [System.Drawing.StringFormat]::GenericDefault)
# Manga shading: a screentone drop shadow behind the lettering...
$tile = New-Object System.Drawing.Bitmap 7, 7
$tg = New-Graphics $tile
$tg.Clear([System.Drawing.Color]::White)
$tg.FillEllipse($inkBrush, 1.6, 1.6, 3.8, 3.8)
$tg.Dispose()
$tone = New-Object System.Drawing.TextureBrush $tile
$shadow = $path.Clone()
$offset = New-Object System.Drawing.Drawing2D.Matrix
$offset.Translate(15, 13)
$shadow.Transform($offset)
$g.DrawPath((New-Object System.Drawing.Pen ([System.Drawing.Color]::White), 18), $shadow)
$g.FillPath($tone, $shadow)
$g.DrawPath((New-Object System.Drawing.Pen $ink, 2), $shadow)
# ...and a graded halftone band rising from the bottom of the title panel.
for ($y = 470; $y -gt 372; $y -= 9) {
    $r = 3.4 * ((470 - $y) / 98.0)
    $r = 3.4 - $r
    for ($x = 14; $x -lt (740 - ($y - 372) * 0.55); $x += 9) {
        if ($r -gt 0.35) { $g.FillEllipse($inkBrush, [float]($x - $r), [float]($y - $r), [float](2 * $r), [float](2 * $r)) }
    }
}
$g.DrawPath((New-Object System.Drawing.Pen ([System.Drawing.Color]::White), 7), $path)
$g.FillPath($inkBrush, $path)
$g.DrawString("TOPOLOGICAL HETEROGENEOUS MANIFOLD ENGINE", $sub, $inkBrush, 104, 284)
$g.DrawString("Take language models apart. Change how they are stored and run.", $small, $inkBrush, 106, 330)
$g.DrawString("Measure what every change costs.", $small, $inkBrush, 106, 360)

# Outer panel frame.
$g.DrawRectangle((New-Object System.Drawing.Pen $ink, 8), 4, 4, $w - 8, $h - 8)
$g.Dispose()
Save-Png $banner "thme-banner.png"
$mascot.Dispose()

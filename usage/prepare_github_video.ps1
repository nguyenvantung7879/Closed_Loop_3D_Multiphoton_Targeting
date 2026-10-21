$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Source = Join-Path (Split-Path -Parent $Root) "fig5\outputs_video\dynamic_quench_11cycles_LF_3D_paired_fast_004137.mp4"
$Output = Join-Path $Root "video\Closed_Loop_3D_Multiphoton_Targeting.mp4"
$PythonExe = "C:\Users\APL\Desktop\SHG-DL\.venv\Scripts\python.exe"

if (!(Test-Path -LiteralPath $Source)) {
    throw "Source video not found: $Source"
}

$Ffmpeg = & $PythonExe -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
if (!(Test-Path -LiteralPath $Ffmpeg)) {
    throw "FFmpeg not found: $Ffmpeg"
}

& $Ffmpeg -y -i $Source -vf "scale=1280:-2" -c:v libx264 -preset medium -crf 24 -pix_fmt yuv420p -movflags +faststart -an $Output
if ($LASTEXITCODE -ne 0) {
    throw "Video conversion failed."
}

Write-Host "Created: $Output"

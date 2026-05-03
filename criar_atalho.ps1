#Requires -Version 5
# Cria o atalho "Passos Mágicos.lnk" com ícone personalizado.
# O ficheiro run.bat nao pode ter icone proprio no Windows (limitacao do sistema).

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ico = Join-Path $root "assets\app.ico"
$bat = Join-Path $root "run.bat"

if (-not (Test-Path $ico)) {
    Write-Error "Nao encontrado: $ico"
    exit 1
}
if (-not (Test-Path $bat)) {
    Write-Error "Nao encontrado: $bat"
    exit 1
}

$WshShell = New-Object -ComObject WScript.Shell
$lnkPath = Join-Path $root "Passos Mágicos.lnk"
$Shortcut = $WshShell.CreateShortcut($lnkPath)
$Shortcut.TargetPath = $bat
$Shortcut.WorkingDirectory = $root
$Shortcut.IconLocation = "$ico,0"
$Shortcut.WindowStyle = 1
$Shortcut.Description = "Passos Mágicos — painel Streamlit local"
$Shortcut.Save()

Write-Host "Atalho criado:" $lnkPath
Write-Host "Arraste para o Ambiente de Trabalho ou use este ficheiro para iniciar (com o icone da imagem)."

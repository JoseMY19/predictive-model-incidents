# Inicia el PostgreSQL dedicado del proyecto (puerto 5433, datos en backend\pgdata). Uso:  .\scripts\db_start.ps1
$bin = (Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\postgres.exe" | Sort-Object FullName -Descending | Select-Object -First 1).DirectoryName
$d = Join-Path $PSScriptRoot "..\pgdata"
if (& "$bin\pg_isready.exe" -h 127.0.0.1 -p 5433) { Write-Host "Ya está corriendo."; exit }
Remove-Item "$d\postmaster.pid" -ErrorAction SilentlyContinue
Start-Process -FilePath "$bin\postgres.exe" -ArgumentList "-D", $d, "-p", "5433", "-c", "listen_addresses=127.0.0.1" `
  -RedirectStandardError "$d\server.log" -RedirectStandardOutput "$env:TEMP\pg_out.log" -WindowStyle Hidden
Start-Sleep 4; & "$bin\pg_isready.exe" -h 127.0.0.1 -p 5433

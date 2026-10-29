# Execute uma vez no PowerShell aberto como Administrador.
# Cria uma tarefa no boot; ela funciona sem login de usuário e inicia novamente
# o arquivo .cmd caso ele seja finalizado pelo Windows ou por falha do processo.

$ErrorActionPreference = 'Stop'
$taskName = 'EcoTrace Backend'
$projectRoot = 'C:\EcoTrace' # Deve coincidir com ECOTRACE_ROOT em run-backend.cmd.
$runner = Join-Path $projectRoot 'web\deploy\windows\run-backend.cmd'

if (-not (Test-Path $runner)) {
    throw "Arquivo nao encontrado: $runner. Copie o repositorio para C:\EcoTrace ou ajuste projectRoot neste script."
}

# O Agendador executa binários de forma mais confiável; cmd.exe inicia o .cmd.
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\cmd.exe" -Argument "/c `"$runner`"" -WorkingDirectory (Split-Path $runner)
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest

Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'EcoTrace FastAPI backend: inicia no boot e reinicia apos falha.' | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State

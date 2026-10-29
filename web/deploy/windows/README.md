# EcoTrace no Windows: inicialização automática

Este método usa o **Agendador de Tarefas do Windows**, não depende de login e
reinicia a API depois de uma reinicialização da máquina. O `run-backend.cmd`
também relança o Uvicorn após uma queda inesperada, aguardando 15 segundos para
evitar um ciclo agressivo de reinicialização.

## 1. Preparar o servidor (uma única vez)

Copie o repositório para `C:\EcoTrace`, instale Python 3.11 de 64 bits e, no
PowerShell, execute:

```powershell
cd C:\EcoTrace\web\backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copie os checkpoints para `C:\EcoTrace\web\backend\models\checkpoints\` e os
GeoTIFFs para `C:\EcoTrace\web\backend\extract-gee\`. O índice
`C:\EcoTrace\web\backend\tile_index.json` deve acompanhar esses arquivos (ou
será criado na primeira inicialização). Se a interface for servida pelo mesmo
backend, gere o frontend uma vez com `npm ci` e `npm run build` dentro de
`web\frontend`.

## 2. Configurar origem e instalar o início automático

Edite `web\deploy\windows\run-backend.cmd` e substitua
`https://SEU-PROJETO.vercel.app` pela URL exata da interface na Vercel. Se não
houver Vercel e a interface for servida pelo próprio backend, essa variável pode
ser removida.

Abra o PowerShell **como Administrador** e execute:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
cd C:\EcoTrace\web\deploy\windows
.\install-autostart.ps1
```

Confirme em `http://IP_DO_SERVIDOR:8000/api/health`. Para checar o serviço e os
registros:

```powershell
Get-ScheduledTask -TaskName 'EcoTrace Backend'
Get-Content C:\EcoTrace\logs\backend.log -Tail 100 -Wait
```

O processo opera com um único worker e `ECOTRACE_PRECACHE=0`, portanto não tenta
carregar os 2.025 GeoTIFFs na inicialização. Isso reduz fortemente o pico de
memória; imagens e prévias são criadas sob demanda e persistidas no cache em
disco.

Para remover a inicialização automática:

```powershell
Unregister-ScheduledTask -TaskName 'EcoTrace Backend' -Confirm:$false
```

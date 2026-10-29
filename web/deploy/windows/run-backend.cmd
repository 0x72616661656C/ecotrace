@echo off
setlocal EnableExtensions

rem Ajuste apenas esta linha se o projeto estiver em outro disco/pasta.
set "ECOTRACE_ROOT=C:\EcoTrace"
set "BACKEND_DIR=%ECOTRACE_ROOT%\web\backend"
set "PYTHON_EXE=%BACKEND_DIR%\.venv\Scripts\python.exe"
set "LOG_DIR=%ECOTRACE_ROOT%\logs"

rem Dados geoespaciais ficam no servidor e não no build do frontend.
set "ECOTRACE_DATA_DIR=%BACKEND_DIR%\extract-gee"
set "ECOTRACE_TILE_INDEX=%BACKEND_DIR%\tile_index.json"

rem A interface hospedada na Vercel pode acessar esta API. Separe origens por vírgula.
set "ECOTRACE_ALLOWED_ORIGINS=https://SEU-PROJETO.vercel.app"

rem Nunca pré-carregue os 2.025 GeoTIFFs neste servidor: isso aumenta muito a RAM no boot.
set "ECOTRACE_PRECACHE=0"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%PYTHON_EXE%" (
  echo [%date% %time%] Python do ambiente virtual nao encontrado: %PYTHON_EXE% >> "%LOG_DIR%\backend.log"
  exit /b 1
)

pushd "%BACKEND_DIR%" || exit /b 1

:restart
echo [%date% %time%] Iniciando EcoTrace API >> "%LOG_DIR%\backend.log"
"%PYTHON_EXE%" -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 >> "%LOG_DIR%\backend.log" 2>&1
echo [%date% %time%] A API parou (codigo %errorlevel%). Tentando novamente em 15 segundos. >> "%LOG_DIR%\backend.log"
timeout /t 15 /nobreak >nul
goto restart

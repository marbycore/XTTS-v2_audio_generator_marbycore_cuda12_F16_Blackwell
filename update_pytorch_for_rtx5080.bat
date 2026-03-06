@echo off
REM Script para actualizar PyTorch a una versión compatible con RTX 5080 (sm_120)
REM Este script instala PyTorch nightly con CUDA 12.8 que soporta arquitectura sm_120

echo ================================================================================
echo Actualizando PyTorch para soporte RTX 5080 (sm_120)
echo ================================================================================
echo.
echo ADVERTENCIA: Esto instalara una version nightly (pre-release) de PyTorch
echo que puede tener bugs o incompatibilidades.
echo.
echo Presiona cualquier tecla para continuar o Ctrl+C para cancelar...
pause >nul

echo.
echo [1/3] Desinstalando PyTorch actual del entorno virtual...
.venv\Scripts\pip.exe uninstall -y torch torchvision torchaudio

echo.
echo [2/3] Instalando PyTorch nightly con CUDA 12.8 (soporte sm_120) en el entorno virtual...
.venv\Scripts\pip.exe install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo.
echo [3/3] Verificando instalacion en el entorno virtual...
.venv\Scripts\python.exe -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No disponible'); cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None; print('CUDA Capability:', f'sm_{cap[0]}{cap[1]}' if cap else 'N/A')"

echo.
echo ================================================================================
echo Actualizacion completada!
echo ================================================================================
echo.
pause


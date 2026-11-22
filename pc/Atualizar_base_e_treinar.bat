@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo == Atualizando listas do QRGBuscador (ILGA + programadas) ==
".venv\Scripts\python.exe" -u qrg_hints.py --update
echo == Baixando novas amostras do sigidwiki ==
del /q data\filelist.json 2>nul
".venv\Scripts\python.exe" -u sigid_download.py --workers 2
echo == Treinando ==
".venv\Scripts\python.exe" -u train.py
pause

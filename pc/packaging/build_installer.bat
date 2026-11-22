@echo off
rem Gera o instalador Windows do SigID Shazam: PyInstaller (pasta) + Inno Setup (setup.exe)
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set ISCC=C:\Users\lualc\BuildTools\InnoSetup\ISCC.exe
if not exist "%ISCC%" set ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe

echo == PyInstaller ==
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed ^
  --name "SigID Shazam" --icon "%CD%\packaging\sigid.ico" ^
  --distpath "%CD%\packaging\dist" --workpath "%CD%\packaging\build" --specpath "%CD%\packaging" ^
  --add-data "%CD%\models\sigid_model.joblib;models" ^
  --add-data "%CD%\data\qrg\ILGADATA.zip;data\qrg" ^
  --add-data "%CD%\data\qrg\programadas.txt;data\qrg" ^
  --hidden-import sklearn.neural_network ^
  --hidden-import sklearn.neighbors._classification ^
  --hidden-import sklearn.decomposition._pca ^
  --hidden-import sklearn.preprocessing._data ^
  --hidden-import sklearn.utils._typedefs ^
  --hidden-import sklearn.neighbors._partition_nodes ^
  --exclude-module tkinter --exclude-module matplotlib --exclude-module IPython ^
  app.py
if errorlevel 1 goto erro

echo == Inno Setup ==
"%ISCC%" "packaging\installer.iss"
if errorlevel 1 goto erro
echo.
echo Instalador pronto em packaging\output
exit /b 0

:erro
echo Falhou.
exit /b 1

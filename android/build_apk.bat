@echo off
rem Exporta o modelo mobile do SigID-Shazam e compila o APK (sem precisar do Android Studio)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set JAVA_HOME=C:\Users\lualc\AndroidBuild\jdk
set ANDROID_HOME=C:\Users\lualc\AndroidBuild\sdk
set GRADLE_USER_HOME=C:\Users\lualc\AndroidBuild\gradle-home

echo == Exportando modelo, metadados e listas QRG ==
"..\pc\.venv\Scripts\python.exe" "..\pc\export_android.py" "..\pc\models\sigid_model_mobile.joblib" "%~dp0."
if errorlevel 1 goto erro

echo == Teste de paridade Kotlin x Python + compilacao ==
call C:\Users\lualc\AndroidBuild\gradle\bin\gradle.bat --no-daemon :app:testDebugUnitTest :app:assembleRelease --console=plain
if errorlevel 1 goto erro

copy /y app\build\outputs\apk\release\app-release.apk SigID-Shazam.apk >nul
echo.
echo APK pronto: %~dp0SigID-Shazam.apk
pause
exit /b 0

:erro
echo Falhou. Veja as mensagens acima.
pause
exit /b 1

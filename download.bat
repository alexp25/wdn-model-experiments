@echo off
echo BAT directory is %~dp0
echo Current directory  is %CD%

del "%~dp0\%data\models\archive.zip"

py -3 -m pip install -r requirements_loader.txt
py -3 download_gdrive.py 1U-Ba2r7wWgwzhDlcWuVCReNL5diT_rEo "%~dp0%\data\models\archive.zip"

pause
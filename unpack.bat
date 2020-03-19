@echo off
echo BAT directory is %~dp0
echo Current directory  is %CD%
pushd %~dp0

del "%~dp0%\archive.zip"
"C:\Program Files\7-Zip\7z.exe" x "%~dp0%\data\models\archive.zip" -o./data/models/unpacked -y -r
pushd %~dp0
pause
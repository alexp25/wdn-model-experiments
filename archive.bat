@echo off
echo BAT directory is %~dp0
echo Current directory  is %CD%
pushd %~dp0

del "%~dp0%\archive.zip"
"C:\Program Files\7-Zip\7z.exe" a -tzip "%~dp0%\data\models\archive.zip" ./data/models/unpacked/*
pushd %~dp0
pause
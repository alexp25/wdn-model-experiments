@echo off
echo BAT directory is %~dp0
echo Current directory  is %CD%
del "%~dp0%\data\models\archive.zip"
"C:\Program Files\7-Zip\7z.exe" a -tzip "%~dp0%\data\models\archive.zip" ./data/models/unpacked/*
pause
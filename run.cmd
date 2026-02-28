@echo off
setlocal

set "ROOT=%~dp0"
powershell -ExecutionPolicy Bypass -File "%ROOT%run.ps1" %*

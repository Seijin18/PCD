@echo off
REM build.bat - helper to run Makefile targets on Windows and provide fallbacks
SETLOCAL ENABLEDELAYEDEXPANSION

REM repository root (this script lives at repo root)
set ROOT=%~dp0
set ENT=%ROOT%Entrega



















































exit /b 1echo Install GNU Make (e.g., from msys2) or run `make` in the `Entrega` folder.echo No fallback implemented for target '%TARGET%'.)    exit /b %ERRORLEVEL%    cl /I "C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Include" "%ENT%\kmeans_1d_mpi.c" /Fe:"%ENT%\bin\kmeans_mpi.exe" /link /LIBPATH:"C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Lib\\x64" msmpi.lib    if not exist "%ENT%\bin" mkdir "%ENT%\bin"    )        exit /b 1        echo vcvarsall.bat failed.    if ERRORLEVEL 1 (    call %VSVC% x64    echo Calling %VSVC% x64 ...    )        exit /b 1        echo Please open a Developer Command Prompt OR install Visual Studio Build Tools.        echo Could not find vcvarsall.bat at common locations.    if not exist %VSVC% (    )        set VSVC="C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat"    if not exist %VSVC% (    set VSVC="C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat"    echo Building MPI binary with MSVC (requires Visual Studio + MS-MPI SDK)...
nif /I "%TARGET%"=="mpi-msvc" ()    exit /b 0    echo Recommended targets: all, serial, omp, cuda, mpi-msvc, clean, clean-results, generate-large, test    echo Usage: build.bat [target]
nif /I "%TARGET%"=="help" (echo Make not found on PATH. Falling back to limited Windows helpers.)    exit /b %ERRORLEVEL%    make -C "%ENT%" %TARGET%    echo Running: make -C "%ENT%" %TARGET%if "%MAKE_FOUND%"=="1" ()    set TARGET=%1) else (    set TARGET=allif "%1"=="" ()    set MAKE_FOUND=0) else (    set MAKE_FOUND=1where make >nul 2>&1
nif %ERRORLEVEL%==0 (nREM detect make
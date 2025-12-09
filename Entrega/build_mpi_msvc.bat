@echo off
REM build_mpi_msvc.bat - bootstrap Visual Studio env and compile MPI binary
setlocal






















exit /b 0echo MPI binary compiled to %~dp0bin\kmeans_mpi.exe)  exit /b 4  echo Compile failedif errorlevel 1 (cl /I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include" "%~dp0kmeans_1d_mpi.c" /Fe:"%~dp0bin\kmeans_mpi.exe" /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib
nif not exist "%~dp0bin" mkdir "%~dp0bin")  exit /b 3  echo vcvarsall failedif errorlevel 1 (
ncall %VCVARS% x64)  exit /b 2  echo Could not find vcvarsall.bat. Please open Developer Command Prompt and run compilation manually.) else (  set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat") else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (  set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"nif exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
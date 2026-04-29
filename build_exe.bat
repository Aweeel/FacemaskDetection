@echo off
setlocal

set "PYTHON=py -3.11"

%PYTHON% -c "import sys; print(sys.version)"
if errorlevel 1 goto :fail

%PYTHON% -m pip install --upgrade pip
if errorlevel 1 goto :fail

%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 goto :fail

%PYTHON% -m PyInstaller --noconfirm --onefile --windowed --name FaceMaskDetection --add-data "web;web" --add-data "models;models" --collect-all tensorflow --collect-all cv2 --collect-all numpy --collect-all setuptools --hidden-import=backports --hidden-import=backports.tarfile --hidden-import=pkg_resources --hidden-import=jaraco --hidden-import=jaraco.context app.py
if errorlevel 1 goto :fail

echo.
echo Build complete. Find the executable in dist\FaceMaskDetection.exe
pause
exit /b 0

:fail
echo.
echo Build failed. Make sure Python 3.11 is installed and available through the py launcher.
pause
exit /b 1
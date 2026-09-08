@echo off
cd /d "G:\My Drive\Project\GLUE01"

git add .

git diff --cached --quiet
if %errorlevel%==0 (
    exit /b 0
)

git commit -m "Auto update %date% %time%"
git push
@echo off
REM ============================================================================
REM AI TEACHING ASSISTANT - WINDOWS LAUNCHER
REM Complete, Production-Ready System
REM ============================================================================

setlocal enabledelayedexpansion
cls

REM Colors
for /F %%A in ('echo prompt $H ^| cmd') do set "BS=%%A"

echo.
echo ╔═══════════════════════════════════════════════════════════╗
echo ║   🎓 AI TEACHING ASSISTANT - PREMIUM LAUNCHER 🎓         ║
echo ║                  Complete ^& Ready to Deploy               ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.

:main_menu
echo.
echo 📊 Quick Stats: Backend [Ready] | Modes [8] | Features [23+]
echo.
echo Choose your option:
echo.
echo 1) 🎨 LAUNCH PREMIUM UI PRO [RECOMMENDED]
echo 2) ⚡ LAUNCH COMPLETE UI
echo 3) 💬 LAUNCH BASIC CHAT UI
echo 4) ⚙️  RUN BACKEND SETUP
echo 5) 📁 OPEN PROJECT FOLDER
echo 6) 📖 OPEN DOCUMENTATION
echo 0) EXIT
echo.

set /p choice="Enter choice (0-6): "

if "%choice%"=="1" goto launch_premium
if "%choice%"=="2" goto launch_complete
if "%choice%"=="3" goto launch_basic
if "%choice%"=="4" goto run_setup
if "%choice%"=="5" goto open_folder
if "%choice%"=="6" goto open_docs
if "%choice%"=="0" goto exit_app
goto invalid_choice

:launch_premium
cls
echo.
echo 🚀 Launching PREMIUM UI Pro...
echo.
echo ✓ Modern design with beautiful gradients
echo ✓ 8 operating modes
echo ✓ Advanced analytics dashboard
echo ✓ Professional settings panel
echo.
echo Starting Streamlit server...
echo 🌐 Opening: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
timeout /t 3
streamlit run app/ui_premium.py
goto main_menu

:launch_complete
cls
echo.
echo ⚡ Launching COMPLETE UI...
echo.
echo ✓ Full-featured interface
echo ✓ 6 core modes
echo ✓ All backend integration
echo ✓ Production-stable
echo.
timeout /t 2
streamlit run app/ui_complete.py
goto main_menu

:launch_basic
cls
echo.
echo 💬 Launching BASIC CHAT UI...
echo.
echo ✓ Simple Q&A interface
echo ✓ RAG system focus
echo ✓ Perfect for testing
echo.
timeout /t 2
streamlit run app/ui.py
goto main_menu

:run_setup
cls
echo.
echo ⚙️  Running Backend Setup...
echo.
python main.py --setup
echo.
pause
goto main_menu

:open_folder
cls
echo.
echo 📁 Opening project folder...
echo.
start .
timeout /t 1
goto main_menu

:open_docs
cls
echo.
echo 📖 Opening README documentation...
echo.
start "" "README.md"
goto main_menu

:invalid_choice
cls
echo.
echo ❌ Invalid choice! Please try again.
echo.
timeout /t 1
goto main_menu

:exit_app
cls
echo.
echo 👋 Thank you for using AI Teaching Assistant PRO!
echo.
echo 📧 Support: support@teachingassistant.ai
echo 🌐 Web: https://teachingassistant.ai
echo.
echo Goodbye! 🚀
echo.
timeout /t 2
exit /b 0

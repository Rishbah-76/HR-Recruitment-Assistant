@echo off
echo HR Recruitment Assistant
echo ========================

REM Check if Ollama is running
curl -s http://localhost:11434/api/version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Ollama server is not running. Please start it first.
    echo Visit https://ollama.com/ for installation instructions.
    set /p continue_choice=Continue without Ollama? (yes/no): 
    if not "%continue_choice%"=="yes" exit /b 1
)

REM Optional: Check if sqlite-vss extension is available
if exist sqlite-vss.dll (
    echo sqlite-vss extension detected
) else (
    echo Warning: sqlite-vss extension not found.
    echo The application will use a fallback similarity method, but it will be slower.
    echo For optimal performance, install sqlite-vss from https://github.com/asg017/sqlite-vss
)

REM Check if models are available (when Ollama is running)
curl -s http://localhost:11434/api/version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Checking required Ollama models...
    
    ollama list | findstr "llama3" >nul
    if %ERRORLEVEL% NEQ 0 (
        echo Model llama3 not found. Pulling...
        ollama pull llama3
    ) else (
        echo llama3 model is available
    )

    ollama list | findstr "nomic-embed-text" >nul
    if %ERRORLEVEL% NEQ 0 (
        echo Model nomic-embed-text not found. Pulling...
        ollama pull nomic-embed-text
    ) else (
        echo nomic-embed-text model is available
    )
)

REM Activate the virtual environment and run the application
call hrapp_env\Scripts\activate
echo Starting the application...
streamlit run app.py 
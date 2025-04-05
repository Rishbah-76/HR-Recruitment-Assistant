#!/bin/bash

echo "HR Recruitment Assistant"
echo "========================"

# Check if Ollama is running
curl -s http://localhost:11434/api/version > /dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Ollama server is not running. Please start it first."
    echo "Visit https://ollama.com/ for installation instructions."
    read -p "Continue without Ollama? (yes/no): " continue_choice
    if [ "$continue_choice" != "yes" ]; then
        exit 1
    fi
fi

# Optional: Check if sqlite-vss extension is available
if [ -f "sqlite-vss.so" ] || [ -f "sqlite-vss.dll" ]; then
    echo "sqlite-vss extension detected"
else
    echo "Warning: sqlite-vss extension not found."
    echo "The application will use a fallback similarity method, but it will be slower."
    echo "For optimal performance, install sqlite-vss from https://github.com/asg017/sqlite-vss"
fi

# Check if models are available (when Ollama is running)
if curl -s http://localhost:11434/api/version > /dev/null; then
    echo "Checking required Ollama models..."
    
    if ! ollama list | grep -q llama3; then
        echo "Model llama3 not found. Pulling..."
        ollama pull llama3
    else
        echo "llama3 model is available"
    fi

    if ! ollama list | grep -q nomic-embed-text; then
        echo "Model nomic-embed-text not found. Pulling..."
        ollama pull nomic-embed-text
    else
        echo "nomic-embed-text model is available"
    fi
fi

# Start the application
echo "Starting the application..."
streamlit run app.py 
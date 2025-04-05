# HR Recruitment Application

An LLM-powered application for processing job descriptions, matching candidate CVs, and scheduling interviews using Ollama and LangChain.

## Overview

This application uses local LLMs through Ollama to automate the recruitment process:

1. **Job Description Processing**: Extract structured information from job descriptions
2. **CV Analysis**: Process candidate CVs and extract relevant information
3. **Matching**: Match candidates with job requirements using semantic similarity
4. **Shortlisting**: Shortlist candidates based on match scores and priority skills
5. **Interview Scheduling**: Generate personalized interview invitation emails

## Technical Stack

- **LLM**: Ollama (llama3 for text generation, nomic-embed-text for embeddings)
- **Framework**: LangChain for agent orchestration
- **Database**: SQLite with vector search extension (sqlite-vss)
- **UI**: Streamlit

## Setup Instructions

### Prerequisites

1. **Install Ollama**:
   - Download from [ollama.com](https://ollama.com/)
   - Start the Ollama server

2. **Install sqlite-vss**:
   - Follow instructions at [github.com/asg017/sqlite-vss](https://github.com/asg017/sqlite-vss)
   - Download the extension for your platform

3. **Install Python dependencies**:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the application**:
   ```
   ./run.sh
   ```
   Or manually:
   ```
   streamlit run app.py
   ```

2. **Load sample data** (optional):
   ```
   python load_samples.py
   ```

## Usage Guide

### 1. Process Job Descriptions

- Enter job title and description text
- The application will use Ollama to extract structured information (skills, qualifications, etc.)
- View the generated JSON summary

### 2. Process Candidate CVs

- Enter candidate name and CV text
- The application will process and store the information

### 3. Match & Shortlist

- Select a job from the dropdown
- Select candidates to match with the job
- View match scores and shortlist candidates based on a threshold
- Adjust rankings using priority skills

### 4. Schedule Interviews

- Select shortlisted candidates
- Generate personalized interview invitation emails

## Project Structure

- `app.py`: Main Streamlit application
- `database.py`: SQLite database with vector search
- `models.py`: Ollama model configurations
- `agents.py`: LangChain agents implementation
- `samples.py`: Sample job descriptions and CVs
- `load_samples.py`: Script to load sample data
- `run.sh`: Bash script to run the application

## How It Works

1. **Job Description Agent**:
   - Uses Ollama to generate structured summaries from job descriptions
   - Stores both the original text and embeddings in the database

2. **CV Processing Agent**:
   - Converts CV text into embeddings
   - Matches CV content with job requirements

3. **Shortlisting Agent**:
   - Filters candidates based on match scores
   - Adjusts rankings based on priority skills

4. **Interview Scheduler Agent**:
   - Generates personalized interview invitation emails

## Benefits

- **Semantic Understanding**: Uses LLMs for deeper understanding of job requirements and candidate qualifications
- **Automation**: Reduces manual screening time
- **Customization**: Easily adjust matching criteria and thresholds
- **Privacy**: All processing happens locally 
# HR Recruitment Application

An LLM-powered application for processing job descriptions, matching candidate CVs, and scheduling interviews using Ollama and LangChain.

## Overview

This application uses local LLMs through Ollama to automate the recruitment process:

1. **Job Description Processing**: Extract structured information from job descriptions
2. **CV Analysis**: Process candidate CVs and extract relevant information
3. **Bulk CV Processing**: Process multiple resumes simultaneously with parallel processing
4. **CV/Resume Enhancement**: Generate improved CVs tailored to specific job requirements
5. **Candidate Matching**: Match candidates with job requirements using semantic similarity
6. **Shortlisting**: Shortlist candidates based on match scores and priority skills
7. **Interview Scheduling**: Generate personalized interview invitation emails
8. **Assessment Test Generation**: Create customized skills assessments for candidates
9. **Analytics Dashboard**: View recruitment metrics and candidate pipeline statistics

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

## Features in Detail

### 1. Process Job Descriptions

- Enter job title and description text
- The application will use Ollama to extract structured information (skills, qualifications, etc.)
- View the generated JSON summary
- Support for bulk job imports from CSV

### 2. Process Candidate CVs

- Enter candidate name and CV text
- Upload PDF resumes with automatic name extraction
- Bulk upload and process multiple candidate CVs simultaneously
- Vector embeddings for semantic matching

### 3. CV/Resume Enhancement

- Generate improved versions of candidate CVs tailored to specific job requirements
- Highlight relevant skills and experience
- Format professional summaries optimized for the target position

### 4. Match & Shortlist

- Select a job from the dropdown
- Select candidates to match with the job
- View match scores and shortlist candidates based on a threshold
- Adjust rankings using priority skills
- Semantic matching beyond keyword matching

### 5. Schedule Interviews

- Select shortlisted candidates
- Generate personalized interview invitation emails
- Include assessment test attachments
- Track email status for each candidate

### 6. Assessment Test Generation

- Create customized skills assessments based on job requirements
- Generate technical questions related to required skills
- Mix of multiple-choice and open-ended questions
- PDF generation for assessment documents

### 7. Analytics Dashboard

- View recruitment pipeline metrics
- Analyze match score distributions
- Track shortlisting rates by job type
- Monitor candidate processing status

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

5. **CV Generation Agent**:
   - Creates tailored CVs optimized for specific job positions

6. **Assessment Generation Agent**:
   - Creates custom skills assessments for candidates

## Benefits

- **Semantic Understanding**: Uses LLMs for deeper understanding of job requirements and candidate qualifications
- **Automation**: Reduces manual screening time
- **Customization**: Easily adjust matching criteria and thresholds
- **Privacy**: All processing happens locally 
- **Bulk Processing**: Handle large volumes of CVs efficiently
- **Data-Driven Decisions**: Analytics dashboard for recruitment insights

[View Demo from Google Drive](https://drive.google.com/file/d/1P8GX_gu8THCZGm4m_d8P7EXUu3VwYqvT/view?usp=sharing)


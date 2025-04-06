import streamlit as st
import json
import os
import tempfile
import asyncio
import time
import sqlite3
import pandas as pd
from database import Database
from models import OllamaModels
from agents import (
    JobDescriptionAgent, 
    CVProcessingAgent, 
    ShortlistingAgent, 
    InterviewSchedulerAgent,
    ResumeProcessingAgent,
    CVGenerationAgent
)

# Initialize database and agents
db = Database()
job_agent = JobDescriptionAgent(db)
cv_agent = CVProcessingAgent(db)
shortlist_agent = ShortlistingAgent(db)
interview_agent = InterviewSchedulerAgent(db)
resume_agent = ResumeProcessingAgent(db)
cv_generation_agent = CVGenerationAgent()

# Set page config
st.set_page_config(
    page_title="HR Recruitment Assistant",
    page_icon="👔",
    layout="wide"
)

# App title
st.title("HR Recruitment Assistant")
st.subheader("Powered by Ollama and LangChain")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:", 
    [
        "Process Job Description", 
        "Process CVs (Bulk)", 
        "CV/Resume Enhancement", 
        "Candidate Matching", 
        "Schedule Interviews",
        "Analytics Dashboard"
    ]
)

# Initialize session state for storing data between pages
if "jobs" not in st.session_state:
    st.session_state.jobs = []
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "matches" not in st.session_state:
    st.session_state.matches = {}  # Job ID -> list of matches
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "bulk_processing_results" not in st.session_state:
    st.session_state.bulk_processing_results = []
if "bulk_match_results" not in st.session_state:
    st.session_state.bulk_match_results = []
if "progress" not in st.session_state:
    st.session_state.progress = {"completed": 0, "total": 0}

# Function to refresh job list
def refresh_jobs():
    cursor = db.conn.cursor()
    cursor.execute("SELECT id, title FROM jobs ORDER BY id DESC")
    st.session_state.jobs = [{"id": row[0], "title": row[1]} for row in cursor.fetchall()]

# Function to refresh candidate list
def refresh_candidates():
    cursor = db.conn.cursor()
    cursor.execute("SELECT id, name FROM candidates ORDER BY id DESC")
    st.session_state.candidates = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]

# Function to refresh match list for a job
def refresh_matches(job_id):
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT m.id, c.id, c.name, m.score, m.is_shortlisted, m.email_sent, m.details "
        "FROM matches m JOIN candidates c ON m.candidate_id = c.id "
        "WHERE m.job_id = ? ORDER BY m.score DESC",
        (job_id,)
    )
    
    st.session_state.matches[job_id] = []
    for row in cursor.fetchall():
        details = {}
        if row[6]:  # details column
            try:
                details = json.loads(row[6])
            except:
                pass
                
        st.session_state.matches[job_id].append({
            "match_id": row[0],
            "candidate_id": row[1],
            "name": row[2],
            "score": row[3],
            "is_shortlisted": bool(row[4]),
            "email_sent": bool(row[5]),
            "details": details
        })

# Helper function to run async code
async def process_resume_async(name, temp_file_path):
    return await resume_agent.process_resume(name, temp_file_path)

def process_resume_sync(name, temp_file_path):
    return asyncio.run(process_resume_async(name, temp_file_path))

# Helper function for bulk processing
async def bulk_process_resumes_async(file_paths):
    return await resume_agent.bulk_process_resumes(file_paths, update_progress_callback)

def bulk_process_resumes_sync(file_paths):
    return asyncio.run(bulk_process_resumes_async(file_paths))

# Add optimized bulk operations
def batch_execute_queries(conn, query, params_list):
    """Execute multiple queries in a single transaction for better performance"""
    cursor = conn.cursor()
    try:
        conn.execute("BEGIN TRANSACTION")
        for params in params_list:
            cursor.execute(query, params)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()

def update_progress_callback(current, total, name=None, status=None):
    """Update progress in session state"""
    st.session_state.progress = {
        "completed": current,
        "total": total,
        "current_name": name,
        "status": status
    }

# Function to get unique job types from the database
def get_job_types():
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT summary FROM jobs WHERE summary IS NOT NULL")
        job_summaries = cursor.fetchall()
        conn.close()
        
        job_types = set()
        for summary_row in job_summaries:
            if summary_row[0]:
                try:
                    summary = json.loads(summary_row[0])
                    if 'job_type' in summary and summary['job_type']:
                        job_types.add(summary['job_type'])
                except json.JSONDecodeError:
                    continue
        
        # Return sorted list of job types with "All" as first option
        return ["All"] + sorted(list(job_types))
    except Exception as e:
        print(f"Error getting job types: {e}")
        return ["All", "Full-time", "Part-time", "Contract", "Internship"]

# Page: Process Job Description
if page == "Process Job Description":
    st.header("Process Job Description")
    
    # Create tabs for single job entry and bulk upload
    job_tab1, job_tab2 = st.tabs(["Single Job Entry", "Bulk Import from CSV"])
    
    with job_tab1:
        # Form to submit job description
        with st.form("job_form"):
            job_title = st.text_input("Job Title", key="job_title")
            job_type = st.text_input("Job Type", value="Full-time", key="job_type")
            job_description = st.text_area("Job Description", height=300, key="job_desc")
            submitted = st.form_submit_button("Process Job")
        
        if submitted and job_title and job_description:
            with st.spinner("Processing job description..."):
                try:
                    job_id = job_agent.process_job_description(job_title, job_description)
                    
                    # Update job type if not already detected in processing
                    conn = sqlite3.connect(db.db_path)
                    c = conn.cursor()
                    c.execute("SELECT summary FROM jobs WHERE id = ?", (job_id,))
                    summary_row = c.fetchone()
                    
                    if summary_row and summary_row[0]:
                        try:
                            summary = json.loads(summary_row[0])
                            if 'job_type' not in summary or not summary['job_type']:
                                summary['job_type'] = job_type
                                c.execute("UPDATE jobs SET summary = ? WHERE id = ?", 
                                         (json.dumps(summary), job_id))
                                conn.commit()
                        except json.JSONDecodeError:
                            pass
                    conn.close()
                    
                    st.success(f"Job processed successfully! Job ID: {job_id}")
                    refresh_jobs()
                    
                    # Display the summary
                    st.subheader("Generated Job Summary")
                    st.json(summary)
                except Exception as e:
                    st.error(f"Error processing job: {str(e)}")
    
    with job_tab2:
        st.subheader("Import Multiple Job Descriptions from CSV")
        
        # Instructions for CSV format
        st.markdown("""
        ### CSV Format Requirements
        Your CSV file should have the following columns:
        - `title`: Job title
        - `description`: Full job description
        
        Additional columns will be ignored.
        """)
        
        # File uploader for CSV
        uploaded_file = st.file_uploader("Upload CSV with job descriptions", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read CSV
                import pandas as pd
                try:
                    # First try UTF-8
                    job_df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    # If that fails, try other common encodings
                    try:
                        # Reset file pointer to beginning
                        uploaded_file.seek(0)
                        job_df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except:
                        # If that fails too, try Windows encoding
                        uploaded_file.seek(0)
                        job_df = pd.read_csv(uploaded_file, encoding='cp1252')
                
                # Check for column name variations and standardize
                column_mapping = {
                    'Job Title': 'title',
                    'Job Description': 'description',
                    'title': 'title',
                    'description': 'description'
                }
                
                # Rename columns if they match expected variations
                for orig_col, standard_col in column_mapping.items():
                    if orig_col in job_df.columns:
                        job_df = job_df.rename(columns={orig_col: standard_col})
                
                # Validate CSV format
                required_columns = ["title", "description"]
                missing_columns = [col for col in required_columns if col not in job_df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}. Please check your CSV format.")
                else:
                    # Display preview
                    st.write("Preview of job descriptions to be imported:")
                    st.dataframe(job_df[["title", "description"]].head())
                    
                    # Process button
                    if st.button("Process All Jobs"):
                        # Setup progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Process each job
                        job_results = []
                        
                        # Preprocess jobs to avoid repetitive work
                        jobs_to_process = []
                        for _, row in job_df.iterrows():
                            jobs_to_process.append({
                                "title": row['title'],
                                "description": row['description']
                            })
                        
                        total_jobs = len(jobs_to_process)
                        status_text.text(f"Preparing to process {total_jobs} jobs...")
                        
                        # Process jobs in batches for better performance
                        batch_size = 5  # Process 5 jobs at a time to balance speed and memory
                        for batch_start in range(0, total_jobs, batch_size):
                            batch_end = min(batch_start + batch_size, total_jobs)
                            current_batch = jobs_to_process[batch_start:batch_end]
                            
                            # Process batch
                            batch_results = []
                            for i, job in enumerate(current_batch):
                                job_index = batch_start + i
                                try:
                                    # Update progress
                                    progress = (job_index + 0.5) / total_jobs
                                    progress_bar.progress(progress)
                                    status_text.text(f"Processing job {job_index+1}/{total_jobs}: {job['title']}")
                                    
                                    # Process job description
                                    job_id = job_agent.process_job_description(job['title'], job['description'])
                                    batch_results.append({
                                        "id": job_id,
                                        "title": job['title'],
                                        "status": "success"
                                    })
                                except Exception as e:
                                    batch_results.append({
                                        "id": None,
                                        "title": job['title'],
                                        "status": f"error: {str(e)}"
                                    })
                            
                            # Update results and progress
                            job_results.extend(batch_results)
                            progress = (batch_end) / total_jobs
                            progress_bar.progress(progress)
                            status_text.text(f"Processed {batch_end}/{total_jobs} jobs")
                        
                        # Complete progress
                        progress_bar.progress(100)
                        status_text.text(f"Processed {total_jobs} job descriptions!")
                        
                        # Refresh jobs list
                        refresh_jobs()
                        
                        # Display results
                        st.subheader("Import Results")
                        results_df = pd.DataFrame(job_results)
                        st.dataframe(results_df)
                        
                        # Count successes and failures
                        success_count = sum(1 for r in job_results if r["status"] == "success")
                        st.success(f"Successfully imported {success_count} out of {len(job_df)} job descriptions.")
                    
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
    
    # Display existing jobs
    st.subheader("Existing Jobs")
    refresh_jobs()
    
    # Show jobs in a more structured way
    if st.session_state.jobs:
        job_data = {
            "ID": [job["id"] for job in st.session_state.jobs],
            "Title": [job["title"] for job in st.session_state.jobs]
        }
        st.dataframe(job_data, use_container_width=True)
        
        # Add option to view job details
        selected_job_id = st.selectbox("Select Job to View Details", 
                                     [f"{job['id']} - {job['title']}" for job in st.session_state.jobs])
        
        if selected_job_id:
            job_id = int(selected_job_id.split(" - ")[0])
            cursor = db.conn.cursor()
            cursor.execute("SELECT title, description, summary FROM jobs WHERE id = ?", (job_id,))
            job_row = cursor.fetchone()
            
            if job_row:
                title, description, summary_json = job_row
                summary = json.loads(summary_json)
                
                st.subheader(f"Job: {title}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Job Description:**")
                    st.text_area("", value=description, height=300, key=f"job_desc_{job_id}", disabled=True)
                
                with col2:
                    st.markdown("**Job Summary:**")
                    st.json(summary)
    else:
        st.info("No jobs in the database yet.")

# Page: Process CVs (Bulk)
elif page == "Process CVs (Bulk)":
    st.header("Process Candidate CVs")
    
    # Add tabs for single vs. bulk processing
    tab1, tab2 = st.tabs(["Bulk CV Processing", "Single CV Processing"])
    
    with tab1:
        st.subheader("Bulk CV Processing")
        
        # Form to submit multiple CVs
        uploaded_files = st.file_uploader("Upload Multiple CVs (PDF)", type=["pdf"], accept_multiple_files=True)
        auto_extract_names = st.checkbox("Automatically extract names from CVs", value=True)
        process_button = st.button("Process CVs in Bulk")
        
        if process_button and uploaded_files:
            # Create temporary files for each uploaded file
            temp_files = []
            try:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_files.append(temp_file.name)
                
                # Setup progress display
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Starting bulk processing...")
                
                # Process files in bulk
                st.session_state.bulk_processing_results = bulk_process_resumes_sync(temp_files)
                
                # Update the progress bar based on completion
                for i in range(len(temp_files)):
                    # Update progress
                    progress = (i + 1) / len(temp_files)
                    progress_bar.progress(progress)
                    
                # Process complete
                progress_bar.progress(100)
                status_text.text(f"Processed {len(temp_files)} CVs successfully!")
                refresh_candidates()
                
                # Display results in a table
                if st.session_state.bulk_processing_results:
                    st.subheader("Processing Results")
                    results_df = {
                        "Name": [],
                        "Status": [],
                        "Candidate ID": []
                    }
                    
                    for result in st.session_state.bulk_processing_results:
                        results_df["Name"].append(result["name"])
                        results_df["Status"].append(result["status"])
                        results_df["Candidate ID"].append(result["candidate_id"] if result["candidate_id"] else "Failed")
                    
                    st.dataframe(results_df)
            
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
    
    with tab2:
        st.subheader("Single CV Processing")
        
        # Form to submit single CV
        with st.form("cv_form"):
            candidate_name = st.text_input("Candidate Name (leave empty for auto-extraction)")
            uploaded_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
            submitted = st.form_submit_button("Process CV")
        
        if submitted and uploaded_file:
            with st.spinner("Processing CV..."):
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                try:
                    # Process the CV
                    name_to_use = candidate_name if candidate_name else "auto"
                    candidate_id, cv_text = process_resume_sync(name_to_use, temp_file_path)
                    st.session_state.resume_text = cv_text
                    
                    # Get the actual name used (for when auto-extracted)
                    cursor = db.conn.cursor()
                    cursor.execute("SELECT name FROM candidates WHERE id = ?", (candidate_id,))
                    actual_name = cursor.fetchone()[0]
                    
                    st.success(f"CV processed successfully! Candidate ID: {candidate_id}")
                    if name_to_use == "auto":
                        st.info(f"Extracted name: {actual_name}")
                    
                    refresh_candidates()
                    
                    # Display CV text
                    with st.expander("CV Content"):
                        st.text(cv_text)
                    
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)
        
        # Option to manually enter CV text
        st.markdown("---")
        st.subheader("Or Enter CV Text Manually")
        with st.form("manual_cv_form"):
            candidate_name_manual = st.text_input("Candidate Name")
            cv_text_manual = st.text_area("CV Text", height=300)
            submitted_manual = st.form_submit_button("Process Manual CV")
        
        if submitted_manual and candidate_name_manual and cv_text_manual:
            with st.spinner("Processing CV..."):
                # Generate embedding for CV
                embedding = OllamaModels.generate_embeddings(cv_text_manual)
                
                # Store in database
                candidate_id = db.add_candidate(candidate_name_manual, cv_text_manual, embedding)
                st.session_state.resume_text = cv_text_manual
                
                st.success(f"CV processed successfully! Candidate ID: {candidate_id}")
                refresh_candidates()
    
    # Display existing candidates
    st.subheader("Existing Candidates")
    refresh_candidates()
    
    if st.session_state.candidates:
        candidates_df = {
            "ID": [c["id"] for c in st.session_state.candidates],
            "Name": [c["name"] for c in st.session_state.candidates]
        }
        st.dataframe(candidates_df, use_container_width=True)
        
        # Add option to view candidate details
        selected_candidate_id = st.selectbox("Select Candidate to View Details", 
                                          [f"{c['id']} - {c['name']}" for c in st.session_state.candidates])
        
        if selected_candidate_id:
            candidate_id = int(selected_candidate_id.split(" - ")[0])
            cursor = db.conn.cursor()
            cursor.execute("SELECT name, cv_text FROM candidates WHERE id = ?", (candidate_id,))
            candidate_row = cursor.fetchone()
            
            if candidate_row:
                name, cv_text = candidate_row
                
                st.subheader(f"Candidate: {name}")
                st.text_area("CV Content", value=cv_text, height=300, key=f"cv_{candidate_id}", disabled=True)
    else:
        st.info("No candidates in the database yet.")

# Page: CV/Resume Enhancement
elif page == "CV/Resume Enhancement":
    st.header("CV/Resume Enhancement")
    
    # Choose resume
    st.subheader("Step 1: Select or Enter Resume Information")
    
    tab1, tab2 = st.tabs(["Select Existing CV", "Enter New Resume Text"])
    
    with tab1:
        refresh_candidates()
        if not st.session_state.candidates:
            st.info("No candidates available. Please process a CV first.")
        else:
            candidate_options = [f"{c['id']} - {c['name']}" for c in st.session_state.candidates]
            selected_candidate = st.selectbox("Select Candidate", candidate_options)
            
            if selected_candidate:
                candidate_id = int(selected_candidate.split(" - ")[0])
                
                # Get resume text
                cursor = db.conn.cursor()
                cursor.execute("SELECT cv_text FROM candidates WHERE id = ?", (candidate_id,))
                resume_row = cursor.fetchone()
                if resume_row:
                    resume_text = resume_row[0]
                    st.session_state.resume_text = resume_text
                    
                    with st.expander("CV Content"):
                        st.text(resume_text)
    
    with tab2:
        new_resume_text = st.text_area("Enter Resume Text", height=300)
        if new_resume_text:
            st.session_state.resume_text = new_resume_text
    
    # Choose job description
    st.subheader("Step 2: Select Job Description")
    
    refresh_jobs()
    if not st.session_state.jobs:
        st.info("No jobs available. Please add a job first.")
    else:
        job_options = [f"{job['id']} - {job['title']}" for job in st.session_state.jobs]
        selected_job = st.selectbox("Select Job", job_options)
        
        if selected_job:
            job_id = int(selected_job.split(" - ")[0])
            
            # Get job description
            cursor = db.conn.cursor()
            cursor.execute("SELECT title, description FROM jobs WHERE id = ?", (job_id,))
            job_row = cursor.fetchone()
            if job_row:
                job_title, job_description = job_row
                
                with st.expander(f"Job Description: {job_title}"):
                    st.text(job_description)
            
                # Generate CV
                if st.button("Generate Tailored CV") and st.session_state.resume_text:
                    with st.spinner("Generating tailored CV..."):
                        cv_text = cv_generation_agent.generate_cv(st.session_state.resume_text, job_description)
                        
                        st.subheader("Generated Tailored CV")
                        generated_cv = st.text_area("CV Content", cv_text, height=500)
                        
                        # Option to save this as a candidate
                        col1, col2 = st.columns(2)
                        with col1:
                            save_name = st.text_input("Enter name for new candidate")
                        with col2:
                            if st.button("Save as New Candidate") and save_name:
                                candidate_id = cv_agent.process_cv(save_name, generated_cv)
                                st.success(f"CV saved as new candidate! Candidate ID: {candidate_id}")
                                refresh_candidates()

# Page: Candidate Matching
elif page == "Candidate Matching":
    st.header("Match Candidates with Jobs")
    
    # Refresh job and candidate lists
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Jobs"):
            refresh_jobs()
    with col2:
        if st.button("Refresh Candidates"):
            refresh_candidates()
    
    # Get job type filter options
    job_types = get_job_types()
    selected_job_type = st.selectbox("Filter Jobs by Type", job_types, index=0)
    
    # Apply job type filter for displayed jobs
    displayed_jobs = st.session_state.jobs
    if selected_job_type != "All":
        displayed_jobs = [
            job for job in st.session_state.jobs 
            if job.get("summary") and isinstance(job.get("summary"), str) and 
            json.loads(job.get("summary", '{}')).get('job_type') == selected_job_type
        ]
    
    # Global match section (match all with all)
    st.markdown("## Global Matching")
    st.markdown("Match all candidates with all jobs in a single operation.")
    
    if st.button("Match All Candidates with All Jobs"):
        if not st.session_state.jobs or not st.session_state.candidates:
            st.warning("Need both jobs and candidates in the database to perform global matching.")
        else:
            # Get all job IDs and candidate IDs - only for the filtered job type if selected
            job_ids = [job["id"] for job in displayed_jobs]
            candidate_ids = [c["id"] for c in st.session_state.candidates]
            
            # Calculate total operations
            total_operations = len(job_ids) * len(candidate_ids)
            
            # Setup progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text(f"Starting global matching: {total_operations} total matches to perform")
            
            # Store results
            match_results = []
            completed_count = 0  # Track global completion
            
            # Process each job
            for job_idx, job_id in enumerate(job_ids):
                job_title = next((job["title"] for job in st.session_state.jobs if job["id"] == job_id), f"Job {job_id}")
                
                # Update status for this job
                status_text.text(f"Processing Job {job_idx+1}/{len(job_ids)}: {job_title}")
                
                # Create a progress tracking function for this job
                def job_progress_callback(completed_candidates, total_candidates):
                    # Access the outer variable directly without nonlocal
                    # since it's defined in the same scope
                    global completed_count 
                    completed_count += 1
                    progress = completed_count / total_operations
                    progress_bar.progress(progress)
                    status_text.text(f"Job {job_idx+1}/{len(job_ids)}: {job_title} - " 
                                   f"Matched {completed_candidates}/{total_candidates} candidates")
                
                # Match all candidates with this job
                job_results = cv_agent.bulk_match_candidates(job_id, candidate_ids, job_progress_callback)
                match_results.append({
                    "job_id": job_id,
                    "job_title": job_title,
                    "results": job_results
                })
            
            # Complete progress
            progress_bar.progress(100)
            status_text.text(f"Global matching complete! Processed {total_operations} matches.")
            
            # Display summary statistics
            st.subheader("Matching Summary")
            
            # Calculate statistics
            jobs_matched = len(match_results)
            candidates_matched = len(candidate_ids)
            total_matches = sum(len(job["results"]) for job in match_results)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jobs Processed", jobs_matched)
            with col2:
                st.metric("Candidates Processed", candidates_matched)
            with col3:
                st.metric("Total Matches Created", total_matches)
                
            # Option to view detailed results
            with st.expander("View Detailed Results", expanded=False):
                for job_result in match_results:
                    st.markdown(f"### {job_result['job_title']}")
                    
                    # Create dataframe for this job's matches
                    job_matches = {
                        "Candidate": [],
                        "Score": [],
                        "Status": []
                    }
                    
                    for match in job_result["results"]:
                        job_matches["Candidate"].append(match["name"])
                        
                        # Format score consistently
                        if "score" in match:
                            score = match["score"]
                            job_matches["Score"].append(f"{score:.2f}" if score > 1 else f"{score*100:.2f}%")
                        else:
                            job_matches["Score"].append("N/A")
                            
                        job_matches["Status"].append(match["status"])
                    
                    st.dataframe(job_matches)
                    st.markdown("---")
    
    st.markdown("---")
    
    # Original job-specific matching
    st.markdown("## Job-Specific Matching")
    
    # Create job options from filtered jobs
    job_options = [f"{job['id']} - {job['title']}" for job in displayed_jobs]
    candidate_options = [f"{c['id']} - {c['name']}" for c in st.session_state.candidates]
    
    if not job_options:
        st.warning(f"No jobs available{' with the selected job type' if selected_job_type != 'All' else ''}. Please add a job first or select a different filter.")
    elif not candidate_options:
        st.warning("No candidates available. Please add candidates first.")
    else:
        # Job selection 
        st.subheader("Step 1: Select Job")
        selected_job = st.selectbox("Select Job", job_options)
        job_id = int(selected_job.split(" - ")[0])
        
        # Get job details
        cursor = db.conn.cursor()
        cursor.execute("SELECT title, summary FROM jobs WHERE id = ?", (job_id,))
        job_row = cursor.fetchone()
        job_title, job_summary_json = job_row
        job_summary = json.loads(job_summary_json)
        
        # Display job summary
        with st.expander(f"Job Summary: {job_title}"):
            st.json(job_summary)
        
        # Add tabs for single vs. bulk matching
        match_tab1, match_tab2 = st.tabs(["Bulk Matching", "Select Individual Candidates"])
        
        with match_tab1:
            st.subheader("Bulk Match All Candidates")
            
            all_candidate_ids = [c["id"] for c in st.session_state.candidates]
            max_to_process = st.slider("Maximum candidates to process", 
                                      min_value=1, 
                                      max_value=len(all_candidate_ids), 
                                      value=min(10, len(all_candidate_ids)))
            
            if st.button("Match All Candidates (Bulk)"):
                # Get subset of candidates to process
                candidates_to_process = all_candidate_ids[:max_to_process]
                
                # Setup progress display
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text(f"Matching {len(candidates_to_process)} candidates with job...")
                
                # Define progress callback
                def update_match_progress(completed, total):
                    progress = completed / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {completed}/{total} candidates")
                
                # Process matching in bulk
                with st.spinner("Matching candidates..."):
                    st.session_state.bulk_match_results = cv_agent.bulk_match_candidates(
                        job_id, candidates_to_process, update_match_progress
                    )
                
                # Show results
                progress_bar.progress(100)
                status_text.text(f"Matching complete for {len(candidates_to_process)} candidates!")
                
                # Refresh matches
                refresh_matches(job_id)
                
                # Display results
                if st.session_state.bulk_match_results:
                    match_results_df = {
                        "Name": [],
                        "Score": [],
                        "Status": []
                    }
                    
                    for result in st.session_state.bulk_match_results:
                        match_results_df["Name"].append(result["name"])
                        
                        if "score" in result:
                            score = result["score"]
                            match_results_df["Score"].append(f"{score:.2f}" if score > 1 else f"{score*100:.2f}%")
                        else:
                            match_results_df["Score"].append("N/A")
                            
                        match_results_df["Status"].append(result["status"])
                    
                    st.dataframe(match_results_df)
        
        with match_tab2:
            st.subheader("Select Individual Candidates to Match")
            
            # Select candidates to match
            selected_candidates = st.multiselect("Select Candidates to Match", candidate_options)
            
            if st.button("Match Selected Candidates") and selected_candidates:
                for candidate_option in selected_candidates:
                    candidate_id = int(candidate_option.split(" - ")[0])
                    candidate_name = candidate_option.split(" - ")[1]
                    
                    with st.spinner(f"Matching {candidate_name} with the job..."):
                        score = cv_agent.match_with_job(job_id, candidate_id)
                        st.success(f"Match completed for {candidate_name}. Match score: {score:.2f}")
                
                # Refresh matches
                refresh_matches(job_id)
        
        # Display matches and shortlisting
        st.markdown("---")
        st.subheader("Matching Results & Shortlisting")
        
        # Refresh matches for selected job
        if job_id not in st.session_state.matches:
            refresh_matches(job_id)
        
        # Display matches
        matches = st.session_state.matches.get(job_id, [])
        if matches:
            # Create DataFrame for visualization with checkboxes
            match_df_data = {
                "Name": [m["name"] for m in matches],
                "Overall Score": [],
                "Direct Score": [],
                "Skills Score": [],
                "Semantic Score": [],
                "Shortlisted": [m["is_shortlisted"] for m in matches]
            }
            
            # Format scores properly
            for m in matches:
                # Overall score 
                overall = m['score']
                # If score is already in percentage format (>1), don't multiply again
                match_df_data["Overall Score"].append(f"{overall:.2f}" if overall > 1 else f"{overall*100:.2f}%")
                
                # Component scores if available
                if m.get('details'):
                    details = m.get('details', {})
                    
                    # Direct score
                    direct = details.get('direct_score', 0)
                    match_df_data["Direct Score"].append(f"{direct:.2f}" if direct > 1 else f"{direct*100:.2f}%")
                    
                    # Skills score
                    skills = details.get('skills_score', 0)
                    match_df_data["Skills Score"].append(f"{skills:.2f}" if skills > 1 else f"{skills*100:.2f}%")
                    
                    # Semantic score
                    semantic = details.get('semantic_score', 0)
                    match_df_data["Semantic Score"].append(f"{semantic:.2f}" if semantic > 1 else f"{semantic*100:.2f}%")
                else:
                    match_df_data["Direct Score"].append("N/A")
                    match_df_data["Skills Score"].append("N/A")
                    match_df_data["Semantic Score"].append("N/A")
            
            st.dataframe(match_df_data)
            
            # Manual candidate selection for shortlisting
            st.markdown("---")
            st.subheader("Select Candidates to Shortlist")
            
            # Create checkboxes for each candidate
            selected_candidates = []
            cols = st.columns(3)  # Display in 3 columns for better spacing
            
            for i, match in enumerate(matches):
                col_idx = i % 3  # Determine which column to place this checkbox
                with cols[col_idx]:
                    is_selected = st.checkbox(
                        f"{match['name']} ({match['score']*100:.2f}% match)" if match['score'] <= 1 
                        else f"{match['name']} ({match['score']:.2f} match)",
                        value=match["is_shortlisted"],
                        key=f"select_{match['match_id']}"
                    )
                    if is_selected:
                        selected_candidates.append(match)
            
            # Shortlist button for manually selected candidates
            if st.button("Shortlist Selected Candidates"):
                if selected_candidates:
                    for match in selected_candidates:
                        # Update database - mark as shortlisted
                        db.update_shortlist(match["match_id"], True)
                    
                    # Confirm and refresh
                    st.success(f"Shortlisted {len(selected_candidates)} candidates")
                    refresh_matches(job_id)
                else:
                    st.warning("No candidates selected for shortlisting")
            
            # Automatic shortlisting with threshold
            st.markdown("---")
            st.subheader("Automatic Shortlisting by Threshold")
            threshold = st.slider("Shortlist Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
            
            if st.button("Apply Threshold"):
                with st.spinner("Shortlisting candidates..."):
                    shortlisted = shortlist_agent.shortlist_candidates(job_id, threshold)
                    st.success(f"Shortlisted {len(shortlisted)} candidates")
                    
                    # Display shortlisted candidates
                    if shortlisted:
                        st.subheader("Shortlisted Candidates")
                        shortlist_df = {
                            "Name": [c["name"] for c in shortlisted],
                            "Score": []
                        }
                        
                        # Format scores properly
                        for c in shortlisted:
                            score = c['score']
                            shortlist_df["Score"].append(f"{score:.2f}" if score > 1 else f"{score*100:.2f}%")
                        
                        st.dataframe(shortlist_df)
                    else:
                        st.info("No candidates met the threshold criteria")
                    
                    # Refresh matches
                    refresh_matches(job_id)
            
            # View detailed match analysis
            st.markdown("---")
            st.subheader("Detailed Match Analysis")
            
            # Format candidate names with scores for selection
            selection_options = []
            for m in matches:
                score = m['score']
                formatted_score = f"{score:.2f}" if score > 1 else f"{score*100:.2f}%"
                selection_options.append(f"{m['name']} ({formatted_score})")
            
            selected_match = st.selectbox(
                "Select Candidate to View Details",
                selection_options
            )
            
            if selected_match:
                selected_name = selected_match.split(" (")[0]
                match = next((m for m in matches if m["name"] == selected_name), None)
                
                if match and match.get("details"):
                    details = match["details"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Match Scores")
                        
                        # Format overall score
                        overall = match['score']
                        st.markdown(f"**Overall Score:** {overall:.2f}" if overall > 1 else f"**Overall Score:** {overall*100:.2f}%")
                        
                        # Format component scores
                        direct = details.get('direct_score', 0)
                        st.markdown(f"**Direct LLM Score:** {direct:.2f}" if direct > 1 else f"**Direct LLM Score:** {direct*100:.2f}%")
                        
                        skills = details.get('skills_score', 0)
                        st.markdown(f"**Skills Match Score:** {skills:.2f}" if skills > 1 else f"**Skills Match Score:** {skills*100:.2f}%")
                        
                        semantic = details.get('semantic_score', 0)
                        st.markdown(f"**Semantic Match Score:** {semantic:.2f}" if semantic > 1 else f"**Semantic Match Score:** {semantic*100:.2f}%")
                    
                    with col2:
                        st.markdown("### Skills Analysis")
                        st.markdown("**Matching Skills:**")
                        matching_skills = details.get("matching_skills", [])
                        for skill in matching_skills:
                            st.markdown(f"- {skill}")
                        
                        st.markdown("**Missing Skills:**")
                        missing_skills = details.get("missing_skills", [])
                        for skill in missing_skills:
                            st.markdown(f"- {skill}")
                        
                        st.markdown("**Matching Preferred Skills:**")
                        matching_preferred = details.get("matching_preferred_skills", [])
                        for skill in matching_preferred:
                            st.markdown(f"- {skill}")
                    
                    st.markdown("### Assessment")
                    st.markdown(details.get("assessment", "No assessment available"))

            # Priority skill ranking
            st.markdown("---")
            st.subheader("Adjust Ranking with Priority Skills")
            priority_skills = st.text_input("Enter priority skills (comma-separated)")

            if st.button("Adjust Ranking") and priority_skills:
                skills_list = [s.strip() for s in priority_skills.split(",")]
                
                with st.spinner("Adjusting candidate ranking..."):
                    adjusted_ranking = shortlist_agent.adjust_ranking(job_id, skills_list)
                    
                    st.subheader("Adjusted Ranking")
                    
                    adjusted_df = {
                        "Name": [c["name"] for c in adjusted_ranking],
                        "Original Score": [],
                        "Adjusted Score": [],
                        "Priority Skills Matched": [", ".join(c["matched_priority_skills"]) for c in adjusted_ranking]
                    }
                    
                    # Format scores properly
                    for c in adjusted_ranking:
                        orig_score = c['original_score']
                        adj_score = c['adjusted_score']
                        
                        adjusted_df["Original Score"].append(f"{orig_score:.2f}" if orig_score > 1 else f"{orig_score*100:.2f}%")
                        adjusted_df["Adjusted Score"].append(f"{adj_score:.2f}" if adj_score > 1 else f"{adj_score*100:.2f}%")
                    
                    st.dataframe(adjusted_df)

        else:
            st.info("No matches for this job yet. Match candidates first.")

# Page: Schedule Interviews
elif page == "Schedule Interviews":
    st.header("Schedule Interviews")
    
    # Select job for scheduling
    refresh_jobs()
    job_options = [f"{job['id']} - {job['title']}" for job in st.session_state.jobs]
    
    if not job_options:
        st.warning("No jobs available. Please add a job first.")
    else:
        selected_job = st.selectbox("Select Job", job_options)
        job_id = int(selected_job.split(" - ")[0])
        
        # Refresh shortlisted candidates
        if job_id not in st.session_state.matches:
            refresh_matches(job_id)
            
        matches = st.session_state.matches.get(job_id, [])
        shortlisted = [m for m in matches if m["is_shortlisted"]]
        
        if not shortlisted:
            st.info("No shortlisted candidates for this job yet. Please shortlist candidates first.")
        else:
            st.subheader("Shortlisted Candidates")
            
            # Interview email settings
            with st.expander("Email Settings", expanded=True):
                company_name = st.text_input("Company Name", "Our Company")
                include_assessment = st.checkbox("Include Skills Assessment Test with Emails", value=True)
                assessment_dir = st.text_input("Assessment Files Directory", "assessments", 
                                             help="Directory where assessment files will be saved")
                
                # Create assessment directory if it doesn't exist
                if not os.path.exists(assessment_dir):
                    try:
                        os.makedirs(assessment_dir)
                        st.success(f"Created assessment directory: {assessment_dir}")
                    except Exception as e:
                        st.error(f"Error creating assessment directory: {str(e)}")
                        include_assessment = False
            
            # Add bulk email option
            bulk_email = st.checkbox("Generate emails for all shortlisted candidates")
            
            if bulk_email:
                if st.button("Generate All Emails"):
                    # Create columns for progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Track assessments
                    assessments_created = []
                    emails_generated = 0
                    errors = []
                    
                    # Get match IDs of candidates that need emails
                    match_ids = [c["match_id"] for c in shortlisted if not c["email_sent"]]
                    
                    if match_ids:
                        status_text.text(f"Generating emails for {len(match_ids)} candidates...")
                        
                        try:
                            # Use the new optimized bulk method
                            results = interview_agent.generate_bulk_emails(match_ids, company_name)
                            
                            # Process results and move assessment files
                            for i, result in enumerate(results):
                                # Update progress
                                progress = (i + 1) / len(results)
                                progress_bar.progress(progress)
                                
                                if result["status"] == "success":
                                    # Get candidate name
                                    candidate = next((c for c in shortlisted if c["match_id"] == result["match_id"]), None)
                                    candidate_name = candidate["name"] if candidate else f"Candidate {result['match_id']}"
                                    
                                    # Store email in session state
                                    key = f"email_{result['match_id']}"
                                    st.session_state[key] = result["email"]
                                    emails_generated += 1
                                    
                                    # Move assessment if created
                                    assessment_path = result.get("assessment_path")
                                    if assessment_path and os.path.exists(assessment_path):
                                        new_path = os.path.join(assessment_dir, os.path.basename(assessment_path))
                                        try:
                                            os.rename(assessment_path, new_path)
                                            assessments_created.append((candidate_name, new_path))
                                        except Exception as e:
                                            errors.append(f"Error moving assessment for {candidate_name}: {str(e)}")
                                else:
                                    errors.append(result["status"])
                                    
                            # Complete progress
                            progress_bar.progress(100)
                            
                        except Exception as e:
                            errors.append(f"Error in bulk processing: {str(e)}")
                            status_text.error(f"Error: {str(e)}")
                            
                        # Refresh matches
                        refresh_matches(job_id)
                    else:
                        status_text.info("No candidates need emails. All selected candidates already have emails sent.")
                        
                    # Report results
                    if emails_generated > 0:
                        status_text.text(f"Generated {emails_generated} out of {len(match_ids)} emails successfully!")
                    elif match_ids:
                        status_text.error("Failed to generate any emails. Please check the errors and try again.")
                    
                    # Show assessment files created
                    if assessments_created:
                        st.subheader("Assessment Files Created")
                        for name, path in assessments_created:
                            st.markdown(f"- **{name}**: [{os.path.basename(path)}]({path})")
                    
                    # Show errors if any
                    if errors:
                        with st.expander(f"Errors ({len(errors)})", expanded=False):
                            for error in errors:
                                st.error(error)
            
            # Display each shortlisted candidate
            for candidate in shortlisted:
                with st.expander(f"{candidate['name']} - Match Score: {candidate['score']:.2f}"):
                    if candidate["email_sent"]:
                        st.success("Email Sent")
                        
                        # Check if we have the email content in session state
                        email_key = f"email_{candidate['match_id']}"
                        if email_key in st.session_state:
                            st.text_area("Email Content", st.session_state[email_key], height=200, key=f"email_display_{candidate['match_id']}")
                        else:
                            st.info("Email was sent but content is not available for display")
                        
                        # Check if there's an assessment file for this candidate
                        try:
                            candidate_assessments = [f for f in os.listdir(assessment_dir) if f.startswith(f"assessment_{candidate['name'].replace(' ', '_')}")]
                            if candidate_assessments:
                                st.success("Assessment Test Generated")
                                for file in candidate_assessments:
                                    file_path = os.path.join(assessment_dir, file)
                                    if os.path.exists(file_path):
                                        with open(file_path, "rb") as f:
                                            st.download_button(
                                                label=f"Download Assessment for {candidate['name']}",
                                                data=f.read(),
                                                file_name=file,
                                                mime="application/pdf",
                                                key=f"download_{candidate['match_id']}"
                                            )
                        except Exception as e:
                            st.warning(f"Could not load assessment files: {str(e)}")
                    else:
                        if st.button(f"Generate Email & Assessment", key=f"gen_email_{candidate['match_id']}"):
                            with st.spinner(f"Generating interview materials for {candidate['name']}..."):
                                try:
                                    # Generate email with assessment if requested
                                    email_content, assessment_path = interview_agent.generate_interview_email(
                                        candidate["match_id"], company_name
                                    )
                                    
                                    # Store in session state
                                    email_key = f"email_{candidate['match_id']}"
                                    st.session_state[email_key] = email_content
                                    
                                    # Display email content
                                    st.subheader("Generated Email")
                                    st.text_area("Email Content", email_content, height=200, key=f"email_content_{candidate['match_id']}")
                                    
                                    # Handle assessment file if created
                                    if assessment_path and os.path.exists(assessment_path):
                                        # Move to specified directory
                                        new_path = os.path.join(assessment_dir, os.path.basename(assessment_path))
                                        try:
                                            os.rename(assessment_path, new_path)
                                            st.success(f"Assessment test created: {os.path.basename(new_path)}")
                                            
                                            # Provide download button
                                            with open(new_path, "rb") as f:
                                                st.download_button(
                                                    label=f"Download Assessment Test",
                                                    data=f.read(),
                                                    file_name=os.path.basename(new_path),
                                                    mime="application/pdf",
                                                    key=f"download_assessment_{candidate['match_id']}"
                                                )
                                        except Exception as e:
                                            st.error(f"Error saving assessment: {str(e)}")
                                    elif include_assessment:
                                        st.warning("Assessment test generation was requested but could not be created. Email was still sent.")
                                
                                except Exception as e:
                                    st.error(f"Error generating interview materials: {str(e)}")
                                    st.info("Try again or contact support if the error persists.")
                                
                                # Refresh matches in any case
                                refresh_matches(job_id)

# Page: Analytics Dashboard
elif page == "Analytics Dashboard":
    st.header("Analytics Dashboard")
    st.subheader("Recruitment Metrics and Insights")
    
    # Fetch data for analytics
    refresh_jobs()
    refresh_candidates()
    
    # Database queries for analytics
    cursor = db.conn.cursor()
    
    # Get overview metrics
    cursor.execute("SELECT COUNT(*) FROM jobs")
    total_jobs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM candidates")
    total_candidates = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM matches")
    total_matches = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM matches WHERE is_shortlisted = 1")
    total_shortlisted = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM matches WHERE email_sent = 1")
    total_interviews = cursor.fetchone()[0]
    
    # KPI metrics section
    st.markdown("### Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="Active Jobs", value=total_jobs)
    with col2:
        st.metric(label="Candidates", value=total_candidates)
    with col3:
        st.metric(label="CV Evaluations", value=total_matches)
    with col4:
        st.metric(label="Shortlisted", value=total_shortlisted)
    with col5:
        st.metric(label="Interviews", value=total_interviews)
    
    # Recruitment funnel
    st.markdown("### Recruitment Funnel")
    funnel_data = {
        "Stage": ["Candidates", "Evaluated", "Shortlisted", "Interviewed"],
        "Count": [total_candidates, total_matches, total_shortlisted, total_interviews]
    }
    
    # Use bar chart for the funnel visualization
    st.bar_chart(funnel_data, x="Stage", y="Count", use_container_width=True)
    
    # Get match score distribution
    cursor.execute("SELECT score FROM matches")
    scores = [row[0] for row in cursor.fetchall()]
    
    if scores:
        st.markdown("### Match Score Distribution")
        
        # Create bins for score distribution
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        
        # Count scores in each bin
        import numpy as np
        hist, _ = np.histogram(scores, bins=bins)
        
        # Create histogram data
        hist_data = {
            "Score Range": labels,
            "Count": hist.tolist()
        }
        
        # Display histogram
        st.bar_chart(hist_data, x="Score Range", y="Count", use_container_width=True)
    
    # Job-specific analytics
    st.markdown("### Job-Specific Analytics")
    
    # Get jobs with candidate counts
    cursor.execute("""
        SELECT j.id, j.title, COUNT(m.id) as candidate_count, 
               SUM(CASE WHEN m.is_shortlisted = 1 THEN 1 ELSE 0 END) as shortlisted_count
        FROM jobs j
        LEFT JOIN matches m ON j.id = m.job_id
        GROUP BY j.id
        ORDER BY candidate_count DESC
    """)
    
    job_stats = cursor.fetchall()
    
    if job_stats:
        # Create job analytics data
        job_data = {
            "Job ID": [row[0] for row in job_stats],
            "Job Title": [row[1] for row in job_stats],
            "Candidates": [row[2] for row in job_stats],
            "Shortlisted": [row[3] for row in job_stats]
        }
        
        # Display job analytics table
        st.dataframe(job_data, use_container_width=True)
        
        # Allow selecting a job for detailed analytics
        job_options = [f"{job[0]} - {job[1]}" for job in job_stats]
        
        if job_options:
            selected_job = st.selectbox("Select Job for Detailed Analytics", job_options)
            job_id = int(selected_job.split(" - ")[0])
            
            # Get match details for the selected job
            if job_id not in st.session_state.matches:
                refresh_matches(job_id)
                
            matches = st.session_state.matches.get(job_id, [])
            
            if matches:
                st.markdown(f"#### Match Score Distribution for {selected_job.split(' - ')[1]}")
                
                # Extract scores
                job_scores = [match["score"] for match in matches]
                
                # Create histogram data
                st.line_chart(job_scores, use_container_width=True)
                
                # Show top candidates
                st.markdown("#### Top Candidates")
                top_candidates = sorted(matches, key=lambda x: x["score"], reverse=True)[:5]
                
                # Format score display
                top_candidate_data = {
                    "Name": [candidate["name"] for candidate in top_candidates],
                    "Score": [f"{candidate['score']*100:.1f}%" for candidate in top_candidates],
                    "Shortlisted": ["Yes" if candidate["is_shortlisted"] else "No" for candidate in top_candidates],
                    "Interview": ["Sent" if candidate["email_sent"] else "Not Sent" for candidate in top_candidates]
                }
                
                st.dataframe(top_candidate_data, use_container_width=True)
                
                # Show detailed match analytics if available
                if any("details" in match and match["details"] for match in matches):
                    st.markdown("#### Skill Match Analytics")
                    
                    # Collect skill match data
                    skills_data = {}
                    
                    for match in matches:
                        if "details" in match and match["details"]:
                            details = match["details"]
                            
                            # Count matching and missing skills
                            matching_skills = details.get("matching_skills", [])
                            missing_skills = details.get("missing_skills", [])
                            
                            for skill in matching_skills:
                                # Handle case where skill might be a dict
                                skill_key = skill if isinstance(skill, str) else str(skill)
                                if skill_key not in skills_data:
                                    skills_data[skill_key] = {"matching": 0, "missing": 0}
                                skills_data[skill_key]["matching"] += 1
                                
                            for skill in missing_skills:
                                # Handle case where skill might be a dict
                                skill_key = skill if isinstance(skill, str) else str(skill)
                                if skill_key not in skills_data:
                                    skills_data[skill_key] = {"matching": 0, "missing": 0}
                                skills_data[skill_key]["missing"] += 1
                    
                    if skills_data:
                        # Create skill match visualization
                        skill_names = list(skills_data.keys())
                        matching_counts = [skills_data[skill]["matching"] for skill in skill_names]
                        missing_counts = [skills_data[skill]["missing"] for skill in skill_names]
                        
                        # Limit to top 10 skills by total mentions
                        if len(skill_names) > 10:
                            skill_totals = [(skill, skills_data[skill]["matching"] + skills_data[skill]["missing"]) 
                                           for skill in skill_names]
                            skill_totals.sort(key=lambda x: x[1], reverse=True)
                            top_skills = [item[0] for item in skill_totals[:10]]
                            
                            # Filter data to top skills
                            skill_names = top_skills
                            matching_counts = [skills_data[skill]["matching"] for skill in skill_names]
                            missing_counts = [skills_data[skill]["missing"] for skill in skill_names]
                        
                        # Create stacked bar chart data
                        skill_chart_data = {
                            "Skill": skill_names * 2,
                            "Count": matching_counts + missing_counts,
                            "Type": ["Matching"] * len(skill_names) + ["Missing"] * len(skill_names)
                        }
                        
                        skill_df = pd.DataFrame(skill_chart_data)
                        
                        # Display skill chart
                        st.bar_chart(skill_df, x="Skill", y="Count", color="Type", use_container_width=True)
    
    # Time-based analytics
    st.markdown("### Recruitment Timeline")
    
    # To simulate timeline data as we don't have actual timestamps
    # In a real implementation, you would add a timestamp to your database tables
    import pandas as pd
    import datetime
    
    # Create a simulated timeline for demonstration
    today = datetime.date.today()
    
    timeline_data = {
        "Date": [
            today - datetime.timedelta(days=30),
            today - datetime.timedelta(days=25),
            today - datetime.timedelta(days=20),
            today - datetime.timedelta(days=15),
            today - datetime.timedelta(days=10),
            today - datetime.timedelta(days=5),
            today
        ],
        "Candidates": [
            total_candidates * 0.1,
            total_candidates * 0.3,
            total_candidates * 0.5,
            total_candidates * 0.7,
            total_candidates * 0.8,
            total_candidates * 0.9,
            total_candidates
        ],
        "Matches": [
            total_matches * 0.1,
            total_matches * 0.3,
            total_matches * 0.5,
            total_matches * 0.7,
            total_matches * 0.8,
            total_matches * 0.9,
            total_matches
        ],
        "Shortlisted": [
            total_shortlisted * 0,
            total_shortlisted * 0.1,
            total_shortlisted * 0.3,
            total_shortlisted * 0.5,
            total_shortlisted * 0.7,
            total_shortlisted * 0.9,
            total_shortlisted
        ],
        "Interviews": [
            total_interviews * 0,
            total_interviews * 0,
            total_interviews * 0.1,
            total_interviews * 0.3,
            total_interviews * 0.5,
            total_interviews * 0.8,
            total_interviews
        ]
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    
    # Display timeline chart
    st.line_chart(timeline_df, x="Date", y=["Candidates", "Matches", "Shortlisted", "Interviews"], use_container_width=True)
    
    # Add efficiency metrics
    if total_candidates > 0 and total_matches > 0:
        st.markdown("### Recruitment Efficiency Metrics")
        
        shortlist_rate = total_shortlisted / total_matches if total_matches > 0 else 0
        interview_rate = total_interviews / total_shortlisted if total_shortlisted > 0 else 0
        
        efficiency_col1, efficiency_col2, efficiency_col3 = st.columns(3)
        
        with efficiency_col1:
            st.metric(label="Match Rate", value=f"{(total_matches/total_candidates)*100:.1f}%")
            
        with efficiency_col2:
            st.metric(label="Shortlist Rate", value=f"{shortlist_rate*100:.1f}%")
            
        with efficiency_col3:
            st.metric(label="Interview Rate", value=f"{interview_rate*100:.1f}%")
    
    # Admin section with password protection
    st.markdown("---")
    with st.expander("⚙️ Admin Controls", expanded=False):
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #007bff;'>
        <h3 style='color: #007bff;'>Database Administration</h3>
        <p>These operations require administrator access. Please enter the admin password to proceed.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Admin password - in a real app, this would be stored securely
        ADMIN_PASSWORD = "hr_admin_2023"  # This is just for demonstration
        
        # Password input
        admin_password = st.text_input("Admin Password", type="password")
        
        if admin_password:
            if admin_password == ADMIN_PASSWORD:
                st.success("Access granted")
                
                # Admin controls
                st.markdown("### Database Operations")
                
                admin_col1, admin_col2 = st.columns(2)
                
                with admin_col1:
                    st.markdown("""
                    <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107;'>
                    <h4 style='color: #856404;'>Reset Database</h4>
                    <p>Clear all matches and evaluations but keep jobs and candidates.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    reset_confirmed = st.checkbox("I understand this will delete all match data")
                    
                    if st.button("Reset Match Data", disabled=not reset_confirmed):
                        try:
                            # Reset database matches but keep jobs and candidates
                            cursor = db.conn.cursor()
                            cursor.execute("DELETE FROM matches")
                            db.conn.commit()
                            
                            # Clear session state matches
                            st.session_state.matches = {}
                            
                            st.success("✅ Successfully reset all match data")
                        except Exception as e:
                            st.error(f"Error resetting database: {str(e)}")
                
                with admin_col2:
                    st.markdown("""
                    <div style='background-color: #f8d7da; padding: 10px; border-radius: 5px; border-left: 5px solid #dc3545;'>
                    <h4 style='color: #721c24;'>Full Database Reset</h4>
                    <p>This will completely reset the database, removing ALL data!</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    delete_confirmed = st.checkbox("I understand this will delete ALL data permanently")
                    delete_double_confirmed = st.checkbox("I confirm this is not a production database")
                    
                    if st.button("Complete Database Reset", 
                                disabled=not (delete_confirmed and delete_double_confirmed),
                                type="primary"):
                        try:
                            # Full database reset
                            cursor = db.conn.cursor()
                            cursor.execute("DELETE FROM matches")
                            cursor.execute("DELETE FROM candidates")
                            cursor.execute("DELETE FROM jobs")
                            db.conn.commit()
                            
                            # Clear session state
                            st.session_state.matches = {}
                            st.session_state.jobs = []
                            st.session_state.candidates = []
                            
                            st.success("✅ Successfully reset the entire database")
                            st.info("You'll need to refresh the page to see the changes in all dashboard sections")
                        except Exception as e:
                            st.error(f"Error resetting database: {str(e)}")
                
                # Database statistics
                st.markdown("### Database Statistics")
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    st.metric("Database Size", f"{os.path.getsize(db.db_path)/1024:.1f} KB")
                    
                with stats_col2:
                    cursor = db.conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                    st.metric("Tables", cursor.fetchone()[0])
                    
                with stats_col3:
                    # Get total rows across all main tables
                    cursor = db.conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM jobs")
                    jobs_count = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) FROM candidates")
                    candidates_count = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) FROM matches")
                    matches_count = cursor.fetchone()[0]
                    
                    st.metric("Total Records", jobs_count + candidates_count + matches_count)
                
                # Export/Backup option
                st.markdown("### Backup & Export")
                if st.button("Create Database Backup"):
                    import shutil
                    import datetime
                    
                    # Create backup with timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = f"hrapp_backup_{timestamp}.db"
                    
                    # Copy the database file
                    shutil.copy2(db.db_path, backup_path)
                    
                    # Provide download link
                    with open(backup_path, "rb") as f:
                        st.download_button(
                            label="Download Backup File",
                            data=f.read(),
                            file_name=backup_path,
                            mime="application/octet-stream"
                        )
            else:
                st.error("Incorrect password. Access denied.")
                
        st.markdown("""
        <div style='font-size: 0.8em; color: #6c757d; margin-top: 15px;'>
        For security reasons, all administrative actions are logged.
        </div>
        """, unsafe_allow_html=True)

# App footer
st.markdown("---")
st.markdown("HR Recruitment Assistant powered by Ollama and LangChain") 
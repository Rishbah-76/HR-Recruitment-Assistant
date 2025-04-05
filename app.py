import streamlit as st
import json
import os
import tempfile
import asyncio
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
        "Process Resumes", 
        "Generate Tailored CV", 
        "Match & Shortlist", 
        "Schedule Interviews"
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
        "SELECT m.id, c.id, c.name, m.score, m.is_shortlisted, m.email_sent "
        "FROM matches m JOIN candidates c ON m.candidate_id = c.id "
        "WHERE m.job_id = ? ORDER BY m.score DESC",
        (job_id,)
    )
    st.session_state.matches[job_id] = [
        {
            "match_id": row[0],
            "candidate_id": row[1],
            "name": row[2],
            "score": row[3],
            "is_shortlisted": bool(row[4]),
            "email_sent": bool(row[5])
        }
        for row in cursor.fetchall()
    ]

# Helper function to run async code
async def process_resume_async(name, temp_file_path):
    return await resume_agent.process_resume(name, temp_file_path)

def process_resume_sync(name, temp_file_path):
    return asyncio.run(process_resume_async(name, temp_file_path))

# Page: Process Job Description
if page == "Process Job Description":
    st.header("Process Job Description")
    
    # Form to submit job description
    with st.form("job_form"):
        job_title = st.text_input("Job Title")
        job_description = st.text_area("Job Description", height=300)
        submitted = st.form_submit_button("Process Job")
    
    if submitted and job_title and job_description:
        with st.spinner("Processing job description..."):
            job_id = job_agent.process_job_description(job_title, job_description)
            
            # Get the summary from the database
            cursor = db.conn.cursor()
            cursor.execute("SELECT summary FROM jobs WHERE id = ?", (job_id,))
            summary_json = cursor.fetchone()[0]
            summary = json.loads(summary_json)
            
            st.success(f"Job processed successfully! Job ID: {job_id}")
            refresh_jobs()
            
            # Display the summary
            st.subheader("Generated Job Summary")
            st.json(summary)
    
    # Display existing jobs
    st.subheader("Existing Jobs")
    refresh_jobs()
    if st.session_state.jobs:
        for job in st.session_state.jobs:
            st.write(f"ID: {job['id']} - {job['title']}")
    else:
        st.info("No jobs in the database yet.")

# Page: Process Resumes
elif page == "Process Resumes":
    st.header("Process Candidate Resumes")
    
    # Form to submit resume
    with st.form("resume_form"):
        candidate_name = st.text_input("Candidate Name")
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        submitted = st.form_submit_button("Process Resume")
    
    if submitted and candidate_name and uploaded_file:
        with st.spinner("Processing resume..."):
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            
            try:
                # Process the resume
                candidate_id, resume_text = process_resume_sync(candidate_name, temp_file_path)
                st.session_state.resume_text = resume_text
                
                st.success(f"Resume processed successfully! Candidate ID: {candidate_id}")
                refresh_candidates()
                
                # Display resume text
                with st.expander("Resume Content"):
                    st.text(resume_text)
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
    
    # Option to manually enter resume text
    st.subheader("Or Enter Resume Text Manually")
    with st.form("manual_resume_form"):
        candidate_name_manual = st.text_input("Candidate Name")
        resume_text_manual = st.text_area("Resume Text", height=300)
        submitted_manual = st.form_submit_button("Process Manual Resume")
    
    if submitted_manual and candidate_name_manual and resume_text_manual:
        with st.spinner("Processing resume..."):
            # Generate embedding for resume
            embedding = OllamaModels.generate_embeddings(resume_text_manual)
            
            # Store in database
            candidate_id = db.add_candidate(candidate_name_manual, resume_text_manual, embedding)
            st.session_state.resume_text = resume_text_manual
            
            st.success(f"Resume processed successfully! Candidate ID: {candidate_id}")
            refresh_candidates()
    
    # Display existing candidates
    st.subheader("Existing Candidates")
    refresh_candidates()
    if st.session_state.candidates:
        for candidate in st.session_state.candidates:
            st.write(f"ID: {candidate['id']} - {candidate['name']}")
    else:
        st.info("No candidates in the database yet.")

# Page: Generate Tailored CV
elif page == "Generate Tailored CV":
    st.header("Generate Tailored CV")
    
    # Choose resume
    st.subheader("Step 1: Select or Enter Resume Information")
    
    tab1, tab2 = st.tabs(["Select Existing Resume", "Enter New Resume"])
    
    with tab1:
        refresh_candidates()
        if not st.session_state.candidates:
            st.info("No candidates available. Please process a resume first.")
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
                    
                    with st.expander("Resume Content"):
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
            cursor.execute("SELECT description FROM jobs WHERE id = ?", (job_id,))
            job_row = cursor.fetchone()
            if job_row:
                job_description = job_row[0]
                
                with st.expander("Job Description"):
                    st.text(job_description)
            
                # Generate CV
                if st.button("Generate Tailored CV") and st.session_state.resume_text:
                    with st.spinner("Generating tailored CV..."):
                        cv_text = cv_generation_agent.generate_cv(st.session_state.resume_text, job_description)
                        
                        st.subheader("Generated Tailored CV")
                        st.text_area("CV Content", cv_text, height=500)
                        
                        # Option to save this as a candidate
                        if st.button("Save as New Candidate"):
                            candidate_name = st.text_input("Enter name for new candidate")
                            if candidate_name:
                                candidate_id = cv_agent.process_cv(candidate_name, cv_text)
                                st.success(f"CV saved as new candidate! Candidate ID: {candidate_id}")
                                refresh_candidates()

# Page: Match & Shortlist
elif page == "Match & Shortlist":
    st.header("Match Candidates with Jobs and Shortlist")
    
    # Refresh job and candidate lists
    refresh_jobs()
    refresh_candidates()
    
    # Select job and candidates for matching
    job_options = [f"{job['id']} - {job['title']}" for job in st.session_state.jobs]
    candidate_options = [f"{c['id']} - {c['name']}" for c in st.session_state.candidates]
    
    if not job_options:
        st.warning("No jobs available. Please add a job first.")
    elif not candidate_options:
        st.warning("No candidates available. Please add candidates first.")
    else:
        # Match tab
        st.subheader("Match Candidates with Job")
        
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
        
        # Shortlist tab
        st.subheader("Shortlist Candidates")
        
        # Refresh matches for selected job
        if job_id not in st.session_state.matches:
            refresh_matches(job_id)
        
        # Display matches
        matches = st.session_state.matches.get(job_id, [])
        if matches:
            match_df_data = {
                "Name": [m["name"] for m in matches],
                "Score": [f"{m['score']:.2f}" for m in matches],
                "Shortlisted": [m["is_shortlisted"] for m in matches]
            }
            
            st.dataframe(match_df_data)
            
            # Shortlist candidates with threshold
            threshold = st.slider("Shortlist Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
            
            if st.button("Shortlist Candidates"):
                with st.spinner("Shortlisting candidates..."):
                    shortlisted = shortlist_agent.shortlist_candidates(job_id, threshold)
                    st.success(f"Shortlisted {len(shortlisted)} candidates")
                    
                    # Display shortlisted candidates
                    if shortlisted:
                        st.subheader("Shortlisted Candidates")
                        for candidate in shortlisted:
                            st.write(f"{candidate['name']} - Score: {candidate['score']:.2f}")
                    else:
                        st.info("No candidates met the threshold criteria")
                    
                    # Refresh matches
                    refresh_matches(job_id)
            
            # Priority skill ranking
            st.subheader("Adjust Ranking with Priority Skills")
            priority_skills = st.text_input("Enter priority skills (comma-separated)")
            
            if st.button("Adjust Ranking") and priority_skills:
                skills_list = [s.strip() for s in priority_skills.split(",")]
                
                with st.spinner("Adjusting candidate ranking..."):
                    adjusted_ranking = shortlist_agent.adjust_ranking(job_id, skills_list)
                    
                    st.subheader("Adjusted Ranking")
                    for candidate in adjusted_ranking:
                        st.write(
                            f"{candidate['name']} - Original Score: {candidate['original_score']:.2f}, "
                            f"Adjusted Score: {candidate['adjusted_score']:.2f}"
                        )
                        
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
            
            for candidate in shortlisted:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{candidate['name']} - Match Score: {candidate['score']:.2f}")
                with col2:
                    if candidate["email_sent"]:
                        st.success("Email Sent")
                    else:
                        if st.button(f"Generate Email for {candidate['name']}", key=f"email_{candidate['match_id']}"):
                            with st.spinner("Generating interview email..."):
                                company_name = st.session_state.get("company_name", "Our Company")
                                email = interview_agent.generate_interview_email(
                                    candidate["match_id"], company_name
                                )
                                
                                st.subheader(f"Email for {candidate['name']}")
                                st.text_area("Email Content", email, height=300, key=f"email_content_{candidate['match_id']}")
                                
                                # Refresh matches
                                refresh_matches(job_id)
            
            # Company name setting
            st.sidebar.subheader("Settings")
            company_name = st.sidebar.text_input("Company Name", "Our Company")
            st.session_state.company_name = company_name

# App footer
st.markdown("---")
st.markdown("HR Recruitment Assistant powered by Ollama and LangChain") 
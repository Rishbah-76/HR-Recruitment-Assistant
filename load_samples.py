#!/usr/bin/env python3
"""
Script to load sample data into the HR Recruitment Assistant database
"""

import time
from database import Database
from models import OllamaModels
from agents import JobDescriptionAgent, CVProcessingAgent
from samples import SAMPLE_JOBS, SAMPLE_CVS

def main():
    """Load sample data into the database"""
    print("Starting to load sample data...")
    
    # Initialize database and agents
    db = Database()
    job_agent = JobDescriptionAgent(db)
    cv_agent = CVProcessingAgent(db)
    
    # Load sample jobs
    print("\nLoading sample jobs...")
    job_ids = []
    for i, job in enumerate(SAMPLE_JOBS):
        print(f"Processing job {i+1}/{len(SAMPLE_JOBS)}: {job['title']}")
        job_id = job_agent.process_job_description(job['title'], job['description'])
        job_ids.append(job_id)
        print(f"Job added with ID: {job_id}")
    
    # Load sample CVs
    print("\nLoading sample CVs...")
    candidate_ids = []
    for i, cv in enumerate(SAMPLE_CVS):
        print(f"Processing CV {i+1}/{len(SAMPLE_CVS)}: {cv['name']}")
        candidate_id = cv_agent.process_cv(cv['name'], cv['cv_text'])
        candidate_ids.append(candidate_id)
        print(f"Candidate added with ID: {candidate_id}")
    
    # Match candidates with jobs
    print("\nMatching candidates with jobs...")
    for job_id in job_ids:
        print(f"\nMatching for job ID: {job_id}")
        cursor = db.conn.cursor()
        cursor.execute("SELECT title FROM jobs WHERE id = ?", (job_id,))
        job_title = cursor.fetchone()[0]
        print(f"Job: {job_title}")
        
        for candidate_id in candidate_ids:
            cursor.execute("SELECT name FROM candidates WHERE id = ?", (candidate_id,))
            candidate_name = cursor.fetchone()[0]
            print(f"Matching with candidate: {candidate_name}...")
            
            score = cv_agent.match_with_job(job_id, candidate_id)
            print(f"Match score: {score:.2f}")
    
    print("\nSample data loading complete!")
    
    # Display summary
    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM jobs")
    jobs_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM candidates")
    candidates_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM matches")
    matches_count = cursor.fetchone()[0]
    
    print(f"\nSummary:")
    print(f"Jobs: {jobs_count}")
    print(f"Candidates: {candidates_count}")
    print(f"Matches: {matches_count}")
    
    db.close()

if __name__ == "__main__":
    main() 
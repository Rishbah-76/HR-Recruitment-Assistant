import json
from typing import Dict, List, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from database import Database
from models import OllamaModels
import re
import asyncio

class JobDescriptionAgent:
    """Agent for processing and summarizing job descriptions"""
    
    def __init__(self, db: Database):
        self.db = db
        self.llm = OllamaModels.get_llm()
    
    def process_job_description(self, title: str, description: str) -> int:
        """Process a job description, generate summary and store in database"""
        # Generate summary using LLM
        prompt = OllamaModels.format_job_summary_prompt(description)
        summary_response = self.llm.invoke(prompt)
        
        # Extract JSON from response
        summary = self._extract_json(summary_response)
        if not summary:
            # Fallback if JSON extraction fails
            summary = {
                "title": title,
                "required_skills": [],
                "preferred_skills": [],
                "qualifications": "",
                "experience": "",
                "responsibilities": [],
                "location": "",
                "job_type": ""
            }
        
        # Generate embedding for the job description
        embedding = OllamaModels.generate_embeddings(description)
        
        # Store in database
        job_id = self.db.add_job(title, description, summary, embedding)
        return job_id
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response"""
        try:
            # Find JSON content with regex
            json_match = re.search(r'({.*})', text.replace('\n', ' '), re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON is found with regex, try to parse the entire text
            return json.loads(text)
        except json.JSONDecodeError:
            return {}


class ResumeProcessingAgent:
    """Agent for processing PDF resumes using langchain"""
    
    def __init__(self, db: Database):
        self.db = db
        self.llm = OllamaModels.get_llm()
    
    async def load_resume_from_pdf(self, file_path: str) -> str:
        """Load resume text from PDF file"""
        loader = PyPDFLoader(file_path)
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        
        # Combine all page content
        resume_text = "\n".join(page.page_content for page in pages)
        return resume_text
    
    async def process_resume(self, name: str, file_path: str) -> int:
        """Process a resume from PDF, generate embedding and store in database"""
        # Load resume text from PDF
        resume_text = await self.load_resume_from_pdf(file_path)
        
        # Generate embedding for resume
        embedding = OllamaModels.generate_embeddings(resume_text)
        
        # Store in database
        candidate_id = self.db.add_candidate(name, resume_text, embedding)
        return candidate_id, resume_text


class CVProcessingAgent:
    """Agent for processing candidate CVs and matching with jobs"""
    
    def __init__(self, db: Database):
        self.db = db
        self.llm = OllamaModels.get_llm()
    
    def process_cv(self, name: str, cv_text: str) -> int:
        """Process a CV, generate embedding and store in database"""
        # Generate embedding for CV
        embedding = OllamaModels.generate_embeddings(cv_text)
        
        # Store in database
        candidate_id = self.db.add_candidate(name, cv_text, embedding)
        return candidate_id
    
    def match_with_job(self, job_id: int, candidate_id: int) -> float:
        """Match a candidate with a job and return match score"""
        # Get job details
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT title, description, summary FROM jobs WHERE id = ?", (job_id,))
        job_row = cursor.fetchone()
        if not job_row:
            return 0.0
        
        job_title, job_description, job_summary_str = job_row
        job_summary = json.loads(job_summary_str)
        
        # Get candidate details
        cursor.execute("SELECT cv_text FROM candidates WHERE id = ?", (candidate_id,))
        candidate_row = cursor.fetchone()
        if not candidate_row:
            return 0.0
        
        cv_text = candidate_row[0]
        
        # Use LLM to match candidate with job
        prompt = OllamaModels.format_candidate_match_prompt(job_summary, cv_text)
        match_response = self.llm.invoke(prompt)
        
        # Extract match score from response
        match_data = self._extract_json(match_response)
        score = float(match_data.get("score", 0)) / 100.0  # Convert percentage to float
        
        # Store match in database
        match_id = self.db.add_match(job_id, candidate_id, score)
        
        return score
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response"""
        try:
            # Find JSON content with regex
            json_match = re.search(r'({.*})', text.replace('\n', ' '), re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON is found with regex, try to parse the entire text
            return json.loads(text)
        except json.JSONDecodeError:
            return {}


class CVGenerationAgent:
    """Agent for generating tailored CVs based on resume and job description"""
    
    def __init__(self):
        self.llm = OllamaModels.get_llm(temperature=0.3)
    
    def generate_cv(self, resume_text: str, job_description: str) -> str:
        """Generate a tailored CV based on resume and job description"""
        prompt = OllamaModels.format_cv_generation_prompt(resume_text, job_description)
        cv_text = self.llm.invoke(prompt)
        return cv_text


class ShortlistingAgent:
    """Agent for shortlisting candidates based on match scores"""
    
    def __init__(self, db: Database):
        self.db = db
        self.llm = OllamaModels.get_llm()
    
    def shortlist_candidates(self, job_id: int, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Shortlist candidates for a job with score above threshold"""
        shortlisted = self.db.get_shortlisted_candidates(job_id, threshold)
        
        # Update shortlist status in database
        for candidate in shortlisted:
            self.db.update_shortlist(candidate["match_id"], True)
        
        return shortlisted
    
    def adjust_ranking(self, job_id: int, priority_skills: List[str]) -> List[Dict[str, Any]]:
        """Adjust candidate ranking based on priority skills"""
        # Get all candidates for the job
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT m.id, c.id, c.name, c.cv_text, m.score FROM matches m "
            "JOIN candidates c ON m.candidate_id = c.id "
            "WHERE m.job_id = ? ORDER BY m.score DESC",
            (job_id,)
        )
        
        candidates = []
        for row in cursor.fetchall():
            match_id, candidate_id, name, cv_text, score = row
            candidates.append({
                "match_id": match_id,
                "candidate_id": candidate_id,
                "name": name,
                "cv_text": cv_text,
                "score": score,
                "adjusted_score": score
            })
        
        # Adjust scores based on priority skills
        for candidate in candidates:
            cv_text = candidate["cv_text"].lower()
            skill_bonus = 0.0
            
            for skill in priority_skills:
                if skill.lower() in cv_text:
                    skill_bonus += 0.05  # 5% bonus per priority skill
            
            candidate["adjusted_score"] = min(1.0, candidate["score"] + skill_bonus)
        
        # Sort by adjusted score
        candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
        
        # Return formatted results
        return [
            {
                "match_id": c["match_id"],
                "candidate_id": c["candidate_id"],
                "name": c["name"],
                "original_score": c["score"],
                "adjusted_score": c["adjusted_score"]
            }
            for c in candidates
        ]


class InterviewSchedulerAgent:
    """Agent for scheduling interviews"""
    
    def __init__(self, db: Database):
        self.db = db
        self.llm = OllamaModels.get_llm(temperature=0.7)  # Higher temperature for creative emails
    
    def generate_interview_email(self, match_id: int, company_name: str = "Our Company") -> str:
        """Generate interview email for a candidate"""
        # Get match details
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT c.name, j.title FROM matches m "
            "JOIN candidates c ON m.candidate_id = c.id "
            "JOIN jobs j ON m.job_id = j.id "
            "WHERE m.id = ?",
            (match_id,)
        )
        
        row = cursor.fetchone()
        if not row:
            return ""
        
        candidate_name, job_title = row
        
        # Generate email using LLM
        prompt = OllamaModels.format_email_prompt(candidate_name, job_title, company_name)
        email = self.llm.invoke(prompt)
        
        # Update email sent status
        self.db.update_email_sent(match_id, True)
        
        return email 
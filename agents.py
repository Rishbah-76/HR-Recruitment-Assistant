import json
from typing import Dict, List, Any, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from database import Database
from models import OllamaModels
import re
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import time

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
    
    async def process_resume(self, name: str, file_path: str) -> Tuple[int, str]:
        """Process a resume from PDF, generate embedding and store in database"""
        # Load resume text from PDF
        resume_text = await self.load_resume_from_pdf(file_path)
        
        # If name is empty or "auto", extract name from resume
        extracted_name = name
        if not name or name.lower() == "auto":
            extracted_name = await self.extract_name_from_resume(resume_text)
            if not extracted_name:
                extracted_name = os.path.basename(file_path)  # Fallback to filename
        
        # Generate embedding for resume
        embedding = OllamaModels.generate_embeddings(resume_text)
        
        # Store in database
        candidate_id = self.db.add_candidate(extracted_name, resume_text, embedding)
        return candidate_id, resume_text
    
    async def extract_name_from_resume(self, resume_text: str) -> str:
        """Extract candidate name from resume text using LLM"""
        prompt = OllamaModels.format_name_extraction_prompt(resume_text)
        response = self.llm.invoke(prompt)
        
        # Try to extract JSON from response
        try:
            # Look for JSON pattern
            json_match = re.search(r'({.*})', response.replace('\n', ' '), re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                if "full_name" in result:
                    return result["full_name"]
            
            # Fallback: Look for direct name mention
            name_match = re.search(r'name[:\s]+([^\n\.]+)', response, re.IGNORECASE)
            if name_match:
                return name_match.group(1).strip()
            
            # Last resort: just return the first line of the response
            return response.strip().split('\n')[0]
        except:
            return "Unknown Candidate"
    
    async def bulk_process_resumes(self, file_paths: List[str], status_callback=None) -> List[Dict[str, Any]]:
        """Process multiple resumes in bulk"""
        results = []
        
        for i, file_path in enumerate(file_paths):
            try:
                # Process resume
                candidate_id, resume_text = await self.process_resume("auto", file_path)
                
                # Get candidate name from database
                cursor = self.db.conn.cursor()
                cursor.execute("SELECT name FROM candidates WHERE id = ?", (candidate_id,))
                name = cursor.fetchone()[0]
                
                results.append({
                    "candidate_id": candidate_id,
                    "name": name,
                    "status": "success",
                    "file_path": file_path
                })
                
                # Update progress if callback provided
                if status_callback:
                    status_callback(i+1, len(file_paths), name, "success")
                    
            except Exception as e:
                results.append({
                    "candidate_id": None,
                    "name": os.path.basename(file_path),
                    "status": f"error: {str(e)}",
                    "file_path": file_path
                })
                
                # Update progress if callback provided
                if status_callback:
                    status_callback(i+1, len(file_paths), os.path.basename(file_path), f"error: {str(e)}")
        
        return results


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
        """Match a candidate with a job using multiple techniques and average the scores"""
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
        
        # Calculate scores using multiple techniques
        scores = []
        
        # Technique 1: LLM direct evaluation
        prompt = OllamaModels.format_candidate_match_prompt(job_summary, cv_text)
        match_response = self.llm.invoke(prompt)
        match_data = self._extract_json(match_response)
        direct_score = float(match_data.get("score", 0)) / 100.0  # Convert percentage to float (0-1 scale)
        scores.append(direct_score)
        
        # Technique 2: Skills matching score
        skills_score = self.calculate_skills_match(job_summary.get("required_skills", []), cv_text)
        scores.append(skills_score)
        
        # Technique 3: Semantic similarity using LLM
        semantic_prompt = OllamaModels.format_semantic_match_prompt(job_description, cv_text)
        semantic_response = self.llm.invoke(semantic_prompt)
        semantic_data = self._extract_json(semantic_response)
        
        # Normalize semantic score to 0-1 scale
        semantic_match = float(semantic_data.get("match_score", 0.5))
        if semantic_match > 1.0:  # If score is on 0-100 scale
            semantic_match = semantic_match / 100.0
        scores.append(semantic_match)
        
        # Calculate the average score
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Calculate the weighted details
        score_details = {
            "direct_score": direct_score,
            "skills_score": skills_score,
            "semantic_score": semantic_match,
            "average_score": avg_score,
            "matching_skills": match_data.get("matching_skills", []),
            "missing_skills": match_data.get("missing_skills", []),
            "matching_preferred_skills": match_data.get("matching_preferred_skills", []),
            "assessment": match_data.get("assessment", "")
        }
        
        # Store match in database with average score
        match_id = self.db.add_match(job_id, candidate_id, avg_score, json.dumps(score_details))
        
        return avg_score
    
    def calculate_skills_match(self, required_skills: List[str], cv_text: str) -> float:
        """Calculate a simple skills match score based on keyword presence"""
        if not required_skills:
            return 0.5  # Neutral score if no required skills
        
        cv_text_lower = cv_text.lower()
        matched_skills = 0
        
        for skill in required_skills:
            # Check for presence of skill in CV text (case-insensitive)
            if skill.lower() in cv_text_lower:
                matched_skills += 1
        
        return matched_skills / len(required_skills) if required_skills else 0.0
    
    def _extract_match_score(self, text: str) -> float:
        """Extract match score from text"""
        # Look for percentage or score mentioned in text
        score_match = re.search(r'(\d+)%|score.*?(\d+(\.\d+)?)', text, re.IGNORECASE)
        if score_match:
            # Try to extract score from different match groups
            if score_match.group(1):  # Percentage format
                return float(score_match.group(1)) / 100.0
            elif score_match.group(2):  # Score format
                score = float(score_match.group(2))
                return min(score / 10.0, 1.0) if score > 1 else score  # Normalize if needed
        
        # No clear score found, default to 0.5
        return 0.5
    
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
    
    def bulk_match_candidates(self, job_id: int, candidate_ids: List[int], status_callback=None) -> List[Dict[str, Any]]:
        """Match multiple candidates with a job in parallel"""
        results = []
        total_candidates = len(candidate_ids)
        
        # Get job info once from the database to avoid multiple queries
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT title, description, summary FROM jobs WHERE id = ?", (job_id,))
        job_row = cursor.fetchone()
        
        if not job_row:
            return []
            
        job_title, job_description, job_summary_str = job_row
        
        # Get all candidate info to avoid repeated queries
        candidates = {}
        cursor.execute("SELECT id, name, cv_text FROM candidates WHERE id IN ({})".format(
            ','.join('?' for _ in candidate_ids)), candidate_ids)
        for row in cursor.fetchall():
            candidates[row[0]] = {"id": row[0], "name": row[1], "cv_text": row[2]}
        
        # Process candidates one by one (no threading)
        completed = 0
        for candidate_id in candidate_ids:
            try:
                # Check if we have the candidate info
                if candidate_id not in candidates:
                    results.append({
                        "candidate_id": candidate_id,
                        "name": f"Candidate {candidate_id}",
                        "score": 0.0,
                        "status": "error: Candidate not found"
                    })
                    completed += 1
                    if status_callback:
                        status_callback(completed, total_candidates)
                    continue
                
                # Get candidate details
                candidate = candidates[candidate_id]
                
                # Calculate scores directly (no DB access in this part)
                # 1. LLM direct evaluation
                prompt = OllamaModels.format_candidate_match_prompt(json.loads(job_summary_str), candidate["cv_text"])
                match_response = self.llm.invoke(prompt)
                match_data = self._extract_json(match_response)
                direct_score = float(match_data.get("score", 0)) / 100.0
                
                # 2. Skills matching score
                skills_score = self.calculate_skills_match(
                    json.loads(job_summary_str).get("required_skills", []), 
                    candidate["cv_text"]
                )
                
                # 3. Semantic matching
                semantic_prompt = OllamaModels.format_semantic_match_prompt(job_description, candidate["cv_text"])
                semantic_response = self.llm.invoke(semantic_prompt)
                semantic_data = self._extract_json(semantic_response)
                semantic_score = float(semantic_data.get("match_score", 0.5))
                
                # Normalize semantic score to 0-1 scale
                if semantic_score > 1.0:  # If score is on 0-100 scale
                    semantic_score = semantic_score / 100.0
                
                # Calculate average
                scores = [direct_score, skills_score, semantic_score]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                # Prepare details
                score_details = {
                    "direct_score": direct_score,
                    "skills_score": skills_score,
                    "semantic_score": semantic_score,
                    "average_score": avg_score,
                    "matching_skills": match_data.get("matching_skills", []),
                    "missing_skills": match_data.get("missing_skills", []),
                    "matching_preferred_skills": match_data.get("matching_preferred_skills", []),
                    "assessment": match_data.get("assessment", "")
                }
                
                # Store match in database
                match_id = self.db.add_match(job_id, candidate_id, avg_score, json.dumps(score_details))
                
                # Add to results
                results.append({
                    "candidate_id": candidate_id,
                    "name": candidate["name"],
                    "score": avg_score,
                    "status": "success"
                })
                
            except Exception as e:
                # Get name if possible
                name = candidates.get(candidate_id, {}).get("name", f"Candidate {candidate_id}")
                results.append({
                    "candidate_id": candidate_id,
                    "name": name,
                    "score": 0.0,
                    "status": f"error: {str(e)}"
                })
                
            # Update progress
            completed += 1
            if status_callback:
                status_callback(completed, total_candidates)
        
        return results


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
            "SELECT m.id, c.id, c.name, c.cv_text, m.score, m.details FROM matches m "
            "JOIN candidates c ON m.candidate_id = c.id "
            "WHERE m.job_id = ? ORDER BY m.score DESC",
            (job_id,)
        )
        
        candidates = []
        for row in cursor.fetchall():
            match_id, candidate_id, name, cv_text, score, details_json = row
            
            # Parse details if available
            details = {}
            if details_json:
                try:
                    details = json.loads(details_json)
                except:
                    pass
                    
            candidates.append({
                "match_id": match_id,
                "candidate_id": candidate_id,
                "name": name,
                "cv_text": cv_text,
                "score": score,
                "details": details,
                "adjusted_score": score
            })
        
        # Adjust scores based on priority skills
        for candidate in candidates:
            cv_text = candidate["cv_text"].lower()
            skill_bonus = 0.0
            matched_priority_skills = []
            
            for skill in priority_skills:
                if skill.lower() in cv_text:
                    skill_bonus += 0.05  # 5% bonus per priority skill
                    matched_priority_skills.append(skill)
            
            candidate["adjusted_score"] = min(1.0, candidate["score"] + skill_bonus)
            candidate["matched_priority_skills"] = matched_priority_skills
        
        # Sort by adjusted score
        candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
        
        # Return formatted results
        return [
            {
                "match_id": c["match_id"],
                "candidate_id": c["candidate_id"],
                "name": c["name"],
                "original_score": c["score"],
                "adjusted_score": c["adjusted_score"],
                "matched_priority_skills": c.get("matched_priority_skills", []),
                "score_details": c.get("details", {})
            }
            for c in candidates
        ]


class TestGenerationAgent:
    """Agent for generating customized assessment tests based on job and candidate profiles"""
    
    def __init__(self):
        self.llm = OllamaModels.get_llm(temperature=0.7)
    
    def generate_assessment_test(self, job_title: str, job_description: str, 
                               candidate_name: str, cv_text: str, match_details: Dict) -> str:
        """
        Generate a customized assessment test based on job and candidate profiles.
        Returns the path to the generated PDF.
        """
        # Create a temporary file for the PDF
        pdf_path = f"assessment_{candidate_name.replace(' ', '_')}_{int(time.time())}.pdf"
        
        try:
            # Generate test content using LLM
            test_content = self._generate_test_content(job_title, job_description, cv_text, match_details)
            
            # Create PDF with the test content
            self._create_test_pdf(pdf_path, job_title, candidate_name, test_content)
            
            # Verify the PDF was created successfully
            if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
                raise ValueError("Failed to create PDF file or file is empty")
            
            return pdf_path
        
        except Exception as e:
            print(f"Error in assessment test generation: {str(e)}")
            # Clean up any partial file that might have been created
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except:
                    pass
            # Re-raise the exception
            raise
    
    def _generate_test_content(self, job_title: str, job_description: str, 
                              cv_text: str, match_details: Dict) -> Dict:
        """Generate the content for the assessment test"""
        prompt = OllamaModels.format_assessment_test_prompt(
            job_title, job_description, cv_text, match_details
        )
        response = self.llm.invoke(prompt)
        
        try:
            # Try to extract JSON content
            test_content = self._extract_json(response)
            return test_content
        except:
            # Fallback to basic structure if JSON parsing fails
            return {
                "introduction": f"Assessment Test for {job_title} position",
                "instructions": "Please complete all questions to the best of your ability.",
                "sections": [
                    {
                        "title": "Technical Assessment",
                        "questions": [
                            {"type": "open", "question": "Please explain your relevant experience for this role."}
                        ]
                    }
                ]
            }
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from text response"""
        try:
            # Find JSON content with regex
            json_match = re.search(r'({.*})', text.replace('\n', ' '), re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON is found with regex, try to parse the entire text
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("Could not parse JSON response")
    
    def _create_test_pdf(self, pdf_path: str, job_title: str, candidate_name: str, 
                         test_content: Dict) -> None:
        """Create a PDF with the assessment test content"""
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'TitleStyle', 
            parent=styles['Heading1'],
            fontSize=16,
            alignment=1,  # Center alignment
            spaceAfter=20
        )
        
        heading_style = ParagraphStyle(
            'HeadingStyle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15
        )
        
        subheading_style = ParagraphStyle(
            'SubheadingStyle',
            parent=styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=10
        )
        
        normal_style = styles['Normal']
        normal_style.fontSize = 11
        normal_style.spaceAfter = 8
        
        # Build the document
        elements = []
        
        # Title
        elements.append(Paragraph(f"Assessment Test: {job_title}", title_style))
        elements.append(Paragraph(f"Candidate: {candidate_name}", heading_style))
        elements.append(Spacer(1, 20))
        
        # Introduction
        if "introduction" in test_content and test_content["introduction"]:
            elements.append(Paragraph("Introduction", heading_style))
            elements.append(Paragraph(str(test_content["introduction"]), normal_style))
            elements.append(Spacer(1, 10))
        
        # Instructions
        if "instructions" in test_content and test_content["instructions"]:
            elements.append(Paragraph("Instructions", heading_style))
            elements.append(Paragraph(str(test_content["instructions"]), normal_style))
            elements.append(Spacer(1, 15))
        
        # Sections with questions
        if "sections" in test_content and isinstance(test_content["sections"], list):
            for section in test_content["sections"]:
                if not isinstance(section, dict) or "title" not in section:
                    continue
                    
                elements.append(Paragraph(str(section["title"]), heading_style))
                
                if "questions" in section and isinstance(section["questions"], list):
                    for i, question in enumerate(section["questions"], 1):
                        if not isinstance(question, dict) or "question" not in question:
                            continue
                            
                        elements.append(Paragraph(f"Q{i}. {str(question['question'])}", subheading_style))
                        
                        # Add answer space depending on question type
                        if question.get("type") == "multiple_choice" and "options" in question and isinstance(question["options"], list):
                            for j, option in enumerate(question["options"], 1):
                                elements.append(Paragraph(f"    {chr(96+j)}) {str(option)}", normal_style))
                        else:
                            # For open questions, add some blank space
                            elements.append(Spacer(1, 40))
                
                elements.append(Spacer(1, 15))
        
        # Build the PDF
        doc.build(elements)


class InterviewSchedulerAgent:
    """Agent for scheduling interviews"""
    
    def __init__(self, db: Database):
        self.db = db
        self.llm = OllamaModels.get_llm(temperature=0.7)  # Higher temperature for creative emails
        self.test_agent = TestGenerationAgent()
    
    def generate_interview_email(self, match_id: int, company_name: str = "Our Company") -> Tuple[str, Optional[str]]:
        """
        Generate interview email for a candidate with an assessment test attached.
        Returns a tuple of (email_content, assessment_test_path).
        """
        # Get match details
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT c.name, j.title, j.description, c.cv_text, m.details "
            "FROM matches m "
            "JOIN candidates c ON m.candidate_id = c.id "
            "JOIN jobs j ON m.job_id = j.id "
            "WHERE m.id = ?",
            (match_id,)
        )
        
        row = cursor.fetchone()
        if not row:
            return "", None
        
        candidate_name, job_title, job_description, cv_text, details_json = row
        
        # Parse match details
        match_details = {}
        if details_json:
            try:
                match_details = json.loads(details_json)
            except:
                pass
        
        # Generate an assessment test
        assessment_path = None
        has_assessment = False
        try:
            assessment_path = self.test_agent.generate_assessment_test(
                job_title, job_description, candidate_name, cv_text, match_details
            )
            has_assessment = True if assessment_path else False
        except Exception as e:
            print(f"Error generating assessment test: {str(e)}")
            has_assessment = False
        
        # Generate email with mention of assessment if one was created
        prompt = OllamaModels.format_email_prompt(
            candidate_name, job_title, company_name, has_assessment
        )
        email = self.llm.invoke(prompt)
        
        # Clean up the email text
        # Remove any introductory text like "Here's a draft email:"
        email_cleaned = email
        intro_patterns = [
            r'^Here\'s a draft email:[\s\n]*',
            r'^Here is a draft email:[\s\n]*',
            r'^Draft email:[\s\n]*',
            r'^I have drafted an email:[\s\n]*',
            r'^Here\'s an email:[\s\n]*',
            r'^Email:[\s\n]*',
            r'^Please find below a professional email:[\s\n]*'
        ]
        
        for pattern in intro_patterns:
            email_cleaned = re.sub(pattern, '', email_cleaned, flags=re.IGNORECASE)
        
        # Update email sent status
        self.db.update_email_sent(match_id, True)
        
        return email_cleaned, assessment_path 
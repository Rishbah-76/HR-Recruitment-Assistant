from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from typing import List, Dict, Any, Optional

class OllamaModels:
    """Class to manage Ollama models for text generation and embeddings"""
    
    @staticmethod
    def get_llm(model_name: str = "llama3", temperature: float = 0.1):
        """Get Ollama LLM for text generation"""
        return OllamaLLM(model=model_name, temperature=temperature)
    
    @staticmethod
    def get_embedding_model(model_name: str = "nomic-embed-text"):
        """Get Ollama embedding model"""
        return OllamaEmbeddings(model=model_name)
    
    @staticmethod
    def generate_embeddings(text: str, model_name: str = "nomic-embed-text") -> List[float]:
        """Generate embeddings for text using Ollama"""
        embeddings = OllamaEmbeddings(model=model_name).embed_query(text)
        return embeddings
    
    @staticmethod
    def format_job_summary_prompt(job_description: str) -> str:
        """Format prompt for job summary generation"""
        return f"""
        You are an expert HR assistant tasked with summarizing job descriptions.
        Please analyze the following job description and extract key information into a structured JSON format.
        
        Include these fields:
        - title: The job title
        - required_skills: List of technical skills required for the position
        - preferred_skills: List of nice-to-have skills
        - qualifications: Educational qualifications required
        - experience: Years of experience required
        - responsibilities: Main job responsibilities
        - location: Work location (if mentioned)
        - job_type: Full-time, part-time, contract, etc.
        
        Format your response as valid JSON only, with no additional text.
        
        Job Description:
        {job_description}
        """
    
    @staticmethod
    def format_candidate_match_prompt(job_summary: Dict[str, Any], cv_text: str) -> str:
        """Format prompt for candidate matching"""
        return f"""
        You are an expert HR recruiter. Compare the following job requirements with the candidate's CV.
        
        Job Requirements:
        {job_summary}
        
        Candidate CV:
        {cv_text}
        
        Provide a detailed analysis of how well the candidate matches the job requirements.
        Include:
        1. A match score from 0-100%
        2. Matching skills from the required skills list
        3. Missing skills from the required skills list
        4. Matching preferred skills
        5. Overall assessment of fit
        
        Format your response as valid JSON only, with these keys: score, matching_skills, missing_skills, matching_preferred_skills, assessment.
        """
    
    @staticmethod
    def format_semantic_match_prompt(job_description: str, cv_text: str) -> str:
        """Format prompt for semantic matching between job and CV"""
        return f"""
        You are an expert HR recruiter assessing the semantic match between a job description and a CV.
        Consider the overall fit, including tone, experience level, and industry alignment.
        
        Job Description:
        {job_description}
        
        Candidate CV:
        {cv_text}
        
        On a scale of 0-100%, how well does this candidate semantically match this position?
        Consider factors beyond just keywords, such as:
        - Experience level match
        - Industry background alignment
        - Seniority level compatibility
        - Overall tone and approach
        
        Provide your response as JSON with the following structure:
        {{
            "match_score": (float between 0 and 1),
            "explanation": "Brief explanation of your assessment"
        }}
        """
    
    @staticmethod
    def format_name_extraction_prompt(resume_text: str) -> str:
        """Format prompt to extract candidate name from resume"""
        # Limit text to first 2000 characters for efficiency
        truncated_text = resume_text[:2000]
        
        return f"""
        You are an expert HR assistant. Extract the full name of the candidate from the following resume text.
        Common resume formats typically have the candidate's name at the top or in a header section.
        
        Resume text (first section):
        {truncated_text}
        
        Return your response in JSON format with the following structure:
        {{
            "full_name": "Extracted full name of the candidate"
        }}
        
        If you cannot determine the full name with confidence, provide your best guess.
        """
    
    @staticmethod
    def format_email_prompt(candidate_name: str, job_title: str, company_name: str = "Our Company", has_assessment: bool = False) -> str:
        """Format prompt for interview email generation"""
        assessment_text = ""
        if has_assessment:
            assessment_text = """
            Also mention that you've attached a skills assessment test that should be completed 
            before the interview. Tell them it's designed to showcase their relevant skills and 
            help guide the interview discussion. Ask them to return the completed assessment 
            at least 24 hours before the interview.
            """
            
        return f"""
        Write a professional email to invite {candidate_name} for an interview for the {job_title} position at {company_name}. 
        Start directly with "Subject:" without any introduction or preamble like "Here's a draft email:".
        
        Include:
        1. A brief introduction about the company
        2. Mention that their qualifications match our requirements
        3. Propose available interview slots (next Monday and Tuesday at 10 AM or 2 PM)
        4. Request confirmation of their preferred slot
        5. Mention that the interview will be conducted via video call
        {assessment_text}
        
        Format as plain text email only. Start directly with the subject line.
        """
    
    @staticmethod
    def format_cv_generation_prompt(resume_text: str, job_description: str) -> str:
        """Format prompt for CV generation"""
        return f"""
        You are an expert career coach and CV writer. Your task is to create a tailored CV for a job applicant 
        based on their resume information and the job description they're applying for.
        
        Resume information:
        {resume_text}
        
        Job Description:
        {job_description}
        
        Please create a professional CV that:
        1. Highlights the most relevant skills and experiences from the resume that match the job requirements
        2. Uses appropriate industry terminology from the job description
        3. Quantifies achievements where possible
        4. Organizes information in a clear, professional format
        5. Is concise and focused on the most relevant qualifications
        
        The CV should include:
        - Professional summary/objective tailored to the position
        - Relevant work experience with accomplishments that match job requirements
        - Skills section highlighting relevant technical and soft skills
        - Education and certifications
        - Any other relevant sections
        
        Format the CV as plain text with clear section headings.
        """
        
    @staticmethod
    def format_assessment_test_prompt(job_title: str, job_description: str, cv_text: str, match_details: Dict) -> str:
        """Format prompt for assessment test generation"""
        # Extract skills data from match details if available
        matching_skills = match_details.get("matching_skills", [])
        missing_skills = match_details.get("missing_skills", [])
        matching_preferred = match_details.get("matching_preferred_skills", [])
        
        skills_info = f"""
        Skills the candidate has shown in their CV:
        {', '.join(matching_skills)}
        
        Skills required for the job that may need assessment:
        {', '.join(missing_skills)}
        
        Additional preferred skills the candidate may have:
        {', '.join(matching_preferred)}
        """
        
        return f"""
        You are an expert HR assessment designer. Create a customized technical assessment test for a candidate 
        applying to the {job_title} position. Design the test to specifically evaluate skills relevant to this role.
        
        Job Description:
        {job_description}
        
        Candidate's CV Summary:
        {cv_text[:500]}
        
        {skills_info}
        
        Create a comprehensive assessment with 2-3 sections covering the most relevant skills for this position.
        Include a mix of multiple-choice and open-ended questions that will challenge the candidate appropriately.
        
        Focus your questions on:
        1. Technical skills specifically required for this role
        2. Areas where the candidate's CV may have gaps compared to requirements
        3. Real-world scenarios the candidate might face in this position
        
        Return your assessment in JSON format with this structure:
        {{
            "introduction": "Brief introduction to the assessment",
            "instructions": "Clear instructions for the candidate",
            "sections": [
                {{
                    "title": "Section title (e.g., Technical Skills, Problem Solving)",
                    "questions": [
                        {{
                            "type": "multiple_choice",
                            "question": "The question text",
                            "options": ["Option 1", "Option 2", "Option 3", "Option 4"]
                        }},
                        {{
                            "type": "open",
                            "question": "An open-ended question"
                        }}
                    ]
                }}
            ]
        }}
        
        Make the assessment challenging but fair, with questions directly relevant to the job requirements.
        """ 
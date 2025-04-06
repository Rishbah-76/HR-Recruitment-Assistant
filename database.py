import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional

class Database:
    def __init__(self, db_path="hrapp.db"):
        self.db_path = db_path
        self.conn = None
        self.has_vss = False
        self.init_db()
        
    def init_db(self):
        """Initialize the database with tables"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Try to load vector extension
        try:
            cursor.execute("SELECT load_extension('sqlite-vss')")
            self.has_vss = True
            print("sqlite-vss extension loaded successfully")
        except sqlite3.OperationalError:
            print("sqlite-vss extension not available, using fallback similarity calculation")
            self.has_vss = False
        
        # Create Jobs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            summary JSON NOT NULL,
            embedding BLOB
        )
        ''')
        
        # Try to create vector index for jobs if vss is available
        if self.has_vss:
            try:
                cursor.execute("SELECT vss_create('jobs', 'embedding', 'cosine')")
                print("Vector index created for jobs table")
            except sqlite3.OperationalError as e:
                print(f"Could not create vector index: {e}")
        
        # Create Candidates table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            cv_text TEXT NOT NULL,
            embedding BLOB
        )
        ''')
        
        # Try to create vector index for candidates if vss is available
        if self.has_vss:
            try:
                cursor.execute("SELECT vss_create('candidates', 'embedding', 'cosine')")
                print("Vector index created for candidates table")
            except sqlite3.OperationalError as e:
                print(f"Could not create vector index: {e}")
        
        # Create job-candidate matches table with details column
        # Check if the matches table already exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='matches'")
        if cursor.fetchone():
            # Check if the details column exists
            cursor.execute("PRAGMA table_info(matches)")
            columns = [info[1] for info in cursor.fetchall()]
            if "details" not in columns:
                # Add details column to existing table
                cursor.execute("ALTER TABLE matches ADD COLUMN details JSON")
                print("Added details column to matches table")
        else:
            # Create new matches table with details column
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                candidate_id INTEGER NOT NULL,
                score REAL NOT NULL,
                is_shortlisted BOOLEAN DEFAULT 0,
                email_sent BOOLEAN DEFAULT 0,
                details JSON,
                FOREIGN KEY (job_id) REFERENCES jobs (id),
                FOREIGN KEY (candidate_id) REFERENCES candidates (id)
            )
            ''')
        
        self.conn.commit()
    
    def _bytes_to_vector(self, embedding_bytes):
        """Convert bytes to numpy array"""
        if isinstance(embedding_bytes, bytes):
            return np.frombuffer(embedding_bytes, dtype=np.float32)
        return np.array(embedding_bytes, dtype=np.float32)
    
    def _vector_to_bytes(self, embedding):
        """Convert vector to bytes for storage"""
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        return embedding.tobytes()
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        vec1 = self._bytes_to_vector(vec1)
        vec2 = self._bytes_to_vector(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)
    
    def add_job(self, title: str, description: str, summary: Dict[str, Any], 
                embedding: List[float]) -> int:
        """Add a job to the database"""
        cursor = self.conn.cursor()
        embedding_bytes = self._vector_to_bytes(embedding)
        cursor.execute(
            "INSERT INTO jobs (title, description, summary, embedding) VALUES (?, ?, ?, ?)",
            (title, description, json.dumps(summary), embedding_bytes)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def add_candidate(self, name: str, cv_text: str, embedding: List[float]) -> int:
        """Add a candidate to the database"""
        cursor = self.conn.cursor()
        embedding_bytes = self._vector_to_bytes(embedding)
        cursor.execute(
            "INSERT INTO candidates (name, cv_text, embedding) VALUES (?, ?, ?)",
            (name, cv_text, embedding_bytes)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def add_match(self, job_id: int, candidate_id: int, score: float, details: str = None) -> int:
        """Record a match between a job and candidate with optional details"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO matches (job_id, candidate_id, score, details) VALUES (?, ?, ?, ?)",
            (job_id, candidate_id, score, details)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def update_shortlist(self, match_id: int, is_shortlisted: bool) -> None:
        """Update shortlist status for a match"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE matches SET is_shortlisted = ? WHERE id = ?",
            (is_shortlisted, match_id)
        )
        self.conn.commit()
    
    def update_email_sent(self, match_id: int, email_sent: bool) -> None:
        """Update email sent status for a match"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE matches SET email_sent = ? WHERE id = ?",
            (email_sent, match_id)
        )
        self.conn.commit()
    
    def find_similar_jobs(self, embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar jobs based on embedding"""
        if self.has_vss:
            # Use sqlite-vss if available
            cursor = self.conn.cursor()
            embedding_bytes = self._vector_to_bytes(embedding)
            cursor.execute(
                "SELECT id, title, summary, vss_cosine_similarity(embedding, ?) as similarity "
                "FROM jobs ORDER BY similarity DESC LIMIT ?",
                (embedding_bytes, limit)
            )
        else:
            # Fallback to manual similarity calculation
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, title, summary, embedding FROM jobs")
            all_jobs = cursor.fetchall()
            
            # Calculate similarity for each job
            job_similarities = []
            for job in all_jobs:
                job_id, title, summary, job_embedding = job
                similarity = self._cosine_similarity(job_embedding, embedding)
                job_similarities.append((job_id, title, summary, similarity))
            
            # Sort by similarity and take top results
            job_similarities.sort(key=lambda x: x[3], reverse=True)
            job_similarities = job_similarities[:limit]
        
        # Format results
        results = []
        if self.has_vss:
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "title": row[1],
                    "summary": json.loads(row[2]),
                    "similarity": row[3]
                })
        else:
            for job_id, title, summary, similarity in job_similarities:
                results.append({
                    "id": job_id,
                    "title": title,
                    "summary": json.loads(summary),
                    "similarity": float(similarity)
                })
                
        return results
    
    def find_similar_candidates(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar candidates based on embedding"""
        if self.has_vss:
            # Use sqlite-vss if available
            cursor = self.conn.cursor()
            embedding_bytes = self._vector_to_bytes(embedding)
            cursor.execute(
                "SELECT id, name, vss_cosine_similarity(embedding, ?) as similarity "
                "FROM candidates ORDER BY similarity DESC LIMIT ?",
                (embedding_bytes, limit)
            )
        else:
            # Fallback to manual similarity calculation
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, name, embedding FROM candidates")
            all_candidates = cursor.fetchall()
            
            # Calculate similarity for each candidate
            candidate_similarities = []
            for candidate in all_candidates:
                candidate_id, name, candidate_embedding = candidate
                similarity = self._cosine_similarity(candidate_embedding, embedding)
                candidate_similarities.append((candidate_id, name, similarity))
            
            # Sort by similarity and take top results
            candidate_similarities.sort(key=lambda x: x[2], reverse=True)
            candidate_similarities = candidate_similarities[:limit]
        
        # Format results
        results = []
        if self.has_vss:
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "name": row[1],
                    "similarity": row[2]
                })
        else:
            for candidate_id, name, similarity in candidate_similarities:
                results.append({
                    "id": candidate_id,
                    "name": name,
                    "similarity": float(similarity)
                })
                
        return results
    
    def get_shortlisted_candidates(self, job_id: int, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Get shortlisted candidates for a job with score above threshold"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT m.id, c.id, c.name, m.score, m.details FROM matches m "
            "JOIN candidates c ON m.candidate_id = c.id "
            "WHERE m.job_id = ? AND m.score >= ? ORDER BY m.score DESC",
            (job_id, threshold)
        )
        results = []
        for row in cursor.fetchall():
            match_id, candidate_id, name, score, details_json = row
            details = {}
            if details_json:
                try:
                    details = json.loads(details_json)
                except:
                    pass
                    
            results.append({
                "match_id": match_id,
                "candidate_id": candidate_id,
                "name": name,
                "score": score,
                "details": details
            })
        return results
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close() 
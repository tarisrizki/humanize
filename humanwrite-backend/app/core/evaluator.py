import sqlite3
import json
import datetime
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Dict, Any

class EvaluationRecord(BaseModel):
    id: Optional[int] = None
    timestamp: Optional[str] = None
    style_mode: str
    language: str
    original_text: str
    output_text: str
    
    # NLP Metrics Comparison
    flesch_original: float
    flesch_output: float
    flesch_target: float
    
    active_voice_original: float
    active_voice_output: float
    active_voice_target: float
    
    # LLM as a Judge Metrics (to be provided by user's prompt system)
    judge_score: Optional[float] = None
    judge_feedback: Optional[str] = None
    
    # Additional context
    metadata: Optional[Dict[str, Any]] = None

class SQLiteEvaluator:
    def __init__(self, db_path: str = None):
        if db_path is None:
            backend_dir = Path(__file__).resolve().parent.parent.parent
            data_dir = backend_dir / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "evaluations.db")
            
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                style_mode TEXT NOT NULL,
                language TEXT NOT NULL,
                original_text TEXT NOT NULL,
                output_text TEXT NOT NULL,
                flesch_original REAL,
                flesch_output REAL,
                flesch_target REAL,
                active_voice_original REAL,
                active_voice_output REAL,
                active_voice_target REAL,
                judge_score REAL,
                judge_feedback TEXT,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_evaluation(self, record: EvaluationRecord) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.datetime.utcnow().isoformat()
        metadata_str = json.dumps(record.metadata) if record.metadata else "{}"
        
        cursor.execute('''
            INSERT INTO evaluations (
                timestamp, style_mode, language, original_text, output_text,
                flesch_original, flesch_output, flesch_target,
                active_voice_original, active_voice_output, active_voice_target,
                judge_score, judge_feedback, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            now,
            record.style_mode,
            record.language,
            record.original_text,
            record.output_text,
            record.flesch_original,
            record.flesch_output,
            record.flesch_target,
            record.active_voice_original,
            record.active_voice_output,
            record.active_voice_target,
            record.judge_score,
            record.judge_feedback,
            metadata_str
        ))
        
        last_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return last_id

    def get_evaluations(self, limit: int = 100, style_mode: str = None) -> list[EvaluationRecord]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM evaluations"
        params = []
        if style_mode:
            query += " WHERE style_mode = ?"
            params.append(style_mode)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            record = EvaluationRecord(
                id=row['id'],
                timestamp=row['timestamp'],
                style_mode=row['style_mode'],
                language=row['language'],
                original_text=row['original_text'],
                output_text=row['output_text'],
                flesch_original=row['flesch_original'],
                flesch_output=row['flesch_output'],
                flesch_target=row['flesch_target'],
                active_voice_original=row['active_voice_original'],
                active_voice_output=row['active_voice_output'],
                active_voice_target=row['active_voice_target'],
                judge_score=row['judge_score'],
                judge_feedback=row['judge_feedback'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            results.append(record)
            
        return results

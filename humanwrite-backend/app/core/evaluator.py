"""Evaluator module for saving and retrieving run history via SQLite."""

import sqlite3
from pathlib import Path
from datetime import datetime
import json
import logging

from app.config import settings

logger = logging.getLogger("humanwrite.evaluator")

DB_PATH = settings.data_path / "evaluations.db"

def _get_db():
    """Create and return a database connection, initializing tables if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    # Initialize table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            style_mode TEXT NOT NULL,
            language TEXT NOT NULL,
            original_text TEXT NOT NULL,
            output_text TEXT NOT NULL,
            trigram_overlap REAL,
            enhanced_judge_score REAL,
            gptzero_enh_human REAL,
            details_json TEXT
        )
    ''')
    conn.commit()
    return conn

def save_evaluation(
    style_mode: str,
    language: str,
    original_text: str,
    output_text: str,
    trigram_overlap: float,
    details: dict = None
) -> int:
    """Save a run to the database and return its ID."""
    conn = _get_db()
    cursor = conn.cursor()
    
    timestamp = datetime.utcnow().isoformat()
    details_str = json.dumps(details) if details else "{}"
    
    cursor.execute('''
        INSERT INTO evaluation_runs 
        (timestamp, style_mode, language, original_text, output_text, trigram_overlap, details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, style_mode, language, original_text, output_text, trigram_overlap, details_str))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return record_id

def get_history(limit: int = 50) -> list[dict]:
    """Retrieve evaluation history."""
    try:
        conn = _get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, style_mode, language, trigram_overlap, enhanced_judge_score, gptzero_enh_human
            FROM evaluation_runs
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        return []

def update_gptzero_score(record_id: int, score: float) -> bool:
    """Update GPTZero score for a specific record."""
    conn = _get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE evaluation_runs 
        SET gptzero_enh_human = ?
        WHERE id = ?
    ''', (score, record_id))
    
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return success

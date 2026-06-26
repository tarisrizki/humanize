import json
import datetime
import re
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseModel

class EvaluationRecord(BaseModel):
    id: Optional[int] = None
    timestamp: Optional[str] = None
    style_mode: str
    language: str
    original_text: str
    output_text: str
    
    # NLP & Review Metrics Comparison
    burstiness: Optional[float] = None
    content_preservation: Optional[float] = None
    ai_word_reduction: Optional[float] = None
    paragraph_integrity: Optional[float] = None
    eyd_score: Optional[float] = None
    
    # LLM as a Judge Meta
    judge_score: Optional[float] = None
    judge_feedback: Optional[str] = None
    
    # GPTZero manual input
    gptzero_before: Optional[float] = None
    gptzero_ai: Optional[int] = None
    gptzero_mixed: Optional[int] = None
    gptzero_human: Optional[int] = None
    
    # Anti-plagiarism
    trigram_overlap: Optional[float] = None
    semantic_similarity: Optional[float] = None
    
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
                burstiness REAL,
                content_preservation REAL,
                ai_word_reduction REAL,
                paragraph_integrity REAL,
                eyd_score REAL,
                judge_score REAL,
                judge_feedback TEXT,
                metadata TEXT,
                gptzero_before REAL,
                gptzero_ai INTEGER,
                gptzero_mixed INTEGER,
                gptzero_human INTEGER,
                trigram_overlap REAL,
                semantic_similarity REAL
            )
        ''')
        
        # Safe migration for existing DB
        try:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN gptzero_before REAL")
            cursor.execute("ALTER TABLE evaluations ADD COLUMN gptzero_after REAL")
        except sqlite3.OperationalError:
            pass # Columns already exist

        try:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN gptzero_ai INTEGER")
            cursor.execute("ALTER TABLE evaluations ADD COLUMN gptzero_mixed INTEGER")
            cursor.execute("ALTER TABLE evaluations ADD COLUMN gptzero_human INTEGER")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN trigram_overlap REAL")
            cursor.execute("ALTER TABLE evaluations ADD COLUMN semantic_similarity REAL")
        except sqlite3.OperationalError:
            pass

            
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
                burstiness, content_preservation, ai_word_reduction,
                paragraph_integrity, eyd_score, judge_score, judge_feedback, metadata,
                gptzero_before, gptzero_after, result_ai, result_mixed, result_human, trigram_overlap, semantic_similarity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            now,
            record.style_mode,
            record.language,
            record.original_text,
            record.output_text,
            record.burstiness,
            record.content_preservation,
            record.ai_word_reduction,
            record.paragraph_integrity,
            record.eyd_score,
            record.judge_score,
            record.judge_feedback,
            metadata_str,
            record.gptzero_before,
            record.gptzero_after, record.result_ai, record.result_mixed, record.result_human, result_ai, result_mixed, result_human,
            record.trigram_overlap,
            record.semantic_similarity
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
                burstiness=row['burstiness'],
                content_preservation=row['content_preservation'],
                ai_word_reduction=row['ai_word_reduction'],
                paragraph_integrity=row['paragraph_integrity'],
                eyd_score=row['eyd_score'],
                judge_score=row['judge_score'],
                judge_feedback=row['judge_feedback'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                gptzero_before=row['gptzero_before'],
                gptzero_ai=row['gptzero_ai'],
                gptzero_mixed=row['gptzero_mixed'],
                gptzero_human=row['gptzero_human'],
                trigram_overlap=row['trigram_overlap'],
                semantic_similarity=row['semantic_similarity']
            )
            results.append(record)
            
        return results

    def update_gptzero_score(self, record_id: int, gptzero_before: float = None, gptzero_ai: int = None, gptzero_mixed: int = None, gptzero_human: int = None) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        params = []
        if gptzero_before is not None:
            updates.append("gptzero_before = ?")
            params.append(gptzero_before)
        if gptzero_ai is not None:
            updates.append("gptzero_ai = ?")
            params.append(gptzero_ai)
        if gptzero_mixed is not None:
            updates.append("gptzero_mixed = ?")
            params.append(gptzero_mixed)
        if gptzero_human is not None:
            updates.append("gptzero_human = ?")
            params.append(gptzero_human)
            
        if not updates:
            conn.close()
            return False
            
        query = f"UPDATE evaluations SET {', '.join(updates)} WHERE id = ?"
        params.append(record_id)
        
        cursor.execute(query, params)
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success

    def update_judge_result(self, record_id: int, judge_score: float, judge_feedback: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE evaluations SET judge_score = ?, judge_feedback = ? WHERE id = ?", 
                       (judge_score, judge_feedback, record_id))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success


JUDGE_PROMPT_TEMPLATE = """
Kamu adalah evaluator ahli untuk sistem humanisasi teks AI.
Tugasmu: menilai apakah teks OUTPUT terdengar seperti ditulis 
manusia, bukan AI, sambil tetap setia pada konten INPUT asli.

═══════════════════════════════════════════════
INPUT ASLI (teks AI-generated yang belum diproses):
{original_text}

OUTPUT HASIL HUMANIZE (yang akan kamu nilai):
{humanized_text}

MODE PENULISAN: {style_mode}
BAHASA: {language}
═══════════════════════════════════════════════

Nilai OUTPUT berdasarkan 5 dimensi ini.
Untuk setiap dimensi, berikan skor 0-10 dan alasan singkat.

DIMENSI 1 — NATURALNESS (seberapa natural terdengar?)
Apakah kalimat-kalimatnya mengalir seperti manusia?
Apakah ada variasi ritme? Apakah tidak terasa robotic?
Skor 0 = sangat AI-like | Skor 10 = sangat natural

DIMENSI 2 — REGISTER COMPLIANCE (sesuai mode yang diminta?)
Mode: {style_mode}
- Akademik: formal, baku, EYD, ada hedging ilmiah
- Profesional: ringkas, to-the-point, aktif, fungsional  
- Populer: mudah dipahami, conversational tapi EYD
- Kreatif: ekspresif, vivid, variasi kalimat ekstrem
Skor 0 = tidak sesuai register | Skor 10 = sempurna

DIMENSI 3 — CONTENT FIDELITY (isi terjaga?)
Apakah semua fakta, angka, makna dari INPUT terjaga?
Skor 0 = banyak fakta berubah | Skor 10 = semua terjaga

DIMENSI 4 — EYD & GRAMMAR (tata bahasa benar?)
Apakah ejaan, tanda baca, struktur kalimat sesuai EYD?
Skor 0 = banyak error EYD | Skor 10 = sempurna

DIMENSI 5 — ANTI-DETECTION (seberapa jauh dari pola AI?)
Apakah ada sisa pola AI: frasa generik, struktur seragam,
transisi klise? Cek: "hal ini menunjukkan", "dapat disimpulkan",
"secara keseluruhan", "sangat penting", "selain itu",
kalimat seragam panjangnya.
Skor 0 = sangat terlihat AI | Skor 10 = tidak ada jejak AI

═══════════════════════════════════════════════
Kembalikan HANYA JSON ini, tanpa teks lain:

{{
  "naturalness":         {{"score": 0, "reason": "..."}},
  "register_compliance": {{"score": 0, "reason": "..."}},
  "content_fidelity":    {{"score": 0, "reason": "..."}},
  "eyd_grammar":         {{"score": 0, "reason": "..."}},
  "anti_detection":      {{"score": 0, "reason": "..."}},
  "overall_score":       0,
  "critical_issues":     [],
  "highlight":           "kalimat paling natural dari output",
  "worst_sentence":      "kalimat yang paling masih terasa AI"
}}

overall_score = (naturalness×25 + register_compliance×20 + 
                 content_fidelity×25 + eyd_grammar×15 + 
                 anti_detection×15) / 10
═══════════════════════════════════════════════
"""

def run_llm_judge(
    original_text: str,
    humanized_text: str,
    style_mode: str,
    language: str,
) -> dict:
    """
    Jalankan LLM as a Judge via Groq API.
    Return dict dengan skor dan feedback.
    """
    import groq
    from app.config import settings
    
    client = groq.Groq(api_key=settings.GROQ_API_KEY)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        original_text=original_text,
        humanized_text=humanized_text,
        style_mode=style_mode,
        language=language,
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            timeout=60.0,
        )

        raw = response.choices[0].message.content.strip()

        # Ekstrak blok JSON dengan aman
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
            
        result = json.loads(raw)

        return {
            "success": True,
            "naturalness":         result.get("naturalness", {"score": 0, "reason": "N/A"}),
            "register_compliance": result.get("register_compliance", {"score": 0, "reason": "N/A"}),
            "content_fidelity":    result.get("content_fidelity", {"score": 0, "reason": "N/A"}),
            "eyd_grammar":         result.get("eyd_grammar", {"score": 0, "reason": "N/A"}),
            "anti_detection":      result.get("anti_detection", {"score": 0, "reason": "N/A"}),
            "overall_score":       result.get("overall_score", 0),
            "critical_issues":     result.get("critical_issues", []),
            "highlight":           result.get("highlight", ""),
            "worst_sentence":      result.get("worst_sentence", ""),
        }

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}", "raw": raw}
    except Exception as e:
        return {"success": False, "error": str(e)}

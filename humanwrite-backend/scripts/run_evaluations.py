import os
import sys
import json
import asyncio
from pathlib import Path

# Setup path so it can run from root or scripts/
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.engine import writing_engine
from app.core.evaluator import SQLiteEvaluator, EvaluationRecord, run_llm_judge

# Inisialisasi Evaluator DB
data_dir = backend_dir / "data"
evaluator = SQLiteEvaluator(db_path=str(data_dir / "evaluations.db"))

TEST_CASES = [
    {
        "mode": "populer",
        "lang": "id",
        "draft": "Secara keseluruhan, tempat wisata ini sangat direkomendasikan. Selain itu, pemandangannya sangat indah dan udaranya segar. Oleh karena itu, Anda harus mengunjunginya."
    },
    {
        "mode": "akademik",
        "lang": "id",
        "draft": "Penelitian ini menunjukkan bahwa terdapat hubungan yang signifikan antara variabel X dan Y. Hal ini dapat disimpulkan bahwa penggunaan metode tersebut sangat efektif untuk meningkatkan hasil."
    },
    {
        "mode": "kreatif",
        "lang": "id",
        "draft": "Di pagi hari yang cerah, burung berkicau dengan merdu. Angin sepoi-sepoi bertiup perlahan. Ia berjalan dengan penuh semangat menuju sekolahnya."
    }
]

async def run_batch_evaluation():
    print("🚀 Memulai Batch Evaluation dengan LLM Judge (Llama-3.3-70b via Groq)...\n")
    
    for i, test in enumerate(TEST_CASES, 1):
        mode = test["mode"]
        lang = test["lang"]
        draft = test["draft"]
        
        print(f"--- TEST CASE {i} [{mode.upper()}] ---")
        print(f"📝 ORIGINAL DRAFT:\n{draft}\n")
        
        print("⚙️  Menjalankan HumanWrite Engine...")
        try:
            # Humanize draft
            output_text, metrics = writing_engine.humanize_text(draft, style_mode=mode)
            print(f"✨ HUMANIZED TEXT:\n{output_text}\n")
            
            # Record ke database untuk dapat ID
            record = EvaluationRecord(
                style_mode=mode,
                language=lang,
                original_text=draft,
                output_text=output_text,
                burstiness=0.0,  # Bisa diisi metrik NLP sebenarnya
                content_preservation=0.0,
                ai_word_reduction=0.0,
                paragraph_integrity=0.0,
                eyd_score=0.0
            )
            record_id = evaluator.log_evaluation(record)
            
            # Jalankan LLM Judge
            print(f"🤖 Menjalankan LLM Judge untuk Record ID: {record_id}...")
            judge_result = run_llm_judge(
                original_text=draft,
                humanized_text=output_text,
                style_mode=mode,
                language=lang
            )
            
            if judge_result.get("success"):
                score = judge_result["overall_score"]
                feedback = {
                    "naturalness": judge_result["naturalness"],
                    "register_compliance": judge_result["register_compliance"],
                    "content_fidelity": judge_result["content_fidelity"],
                    "eyd_grammar": judge_result["eyd_grammar"],
                    "anti_detection": judge_result["anti_detection"],
                    "critical_issues": judge_result["critical_issues"],
                    "highlight": judge_result["highlight"],
                    "worst_sentence": judge_result["worst_sentence"]
                }
                
                # Simpan skor dan feedback ke DB
                evaluator.update_judge_result(
                    record_id=record_id,
                    judge_score=score,
                    judge_feedback=json.dumps(feedback, ensure_ascii=False)
                )
                
                print(f"✅ Judge Score: {score}/100")
                if feedback['critical_issues']:
                    print(f"⚠️  Critical Issues: {feedback['critical_issues']}")
                print(f"🌟 Highlight: {feedback['highlight']}")
            else:
                print(f"❌ LLM Judge Error: {judge_result.get('error')}")
                
        except Exception as e:
            print(f"❌ Error memproses test case {i}: {e}")
            
        print("-" * 50 + "\n")
        
    print("🎯 Batch Evaluation Selesai! Anda bisa melihat history di SQLite evaluations.db")

if __name__ == "__main__":
    asyncio.run(run_batch_evaluation())

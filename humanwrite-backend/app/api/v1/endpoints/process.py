"""Process endpoint — rewrites a draft to match the user's writing style."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.core.writing_engine import apply_style, apply_style_stream
from app.models.schemas import ProcessedText, ProcessRequest
from app.models.style_profile import StyleProfile
from app.storage.json_store import load_json, save_json

router = APIRouter()


@router.get("/style", response_model=StyleProfile)
async def get_global_style(mode: str = "populer") -> StyleProfile:
    """Return the pre-trained Global StyleProfile."""
    profile_path = settings.profiles_path / f"{mode}_style.json"
    if not profile_path.exists():
        profile_path = settings.profiles_path / "global_style.json"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="Style profile not found.")
    data = load_json(profile_path)
    data["style_mode"] = mode
    return StyleProfile(**data)


@router.post("/process")
async def process_draft(request: ProcessRequest) -> StreamingResponse:
    """Rewrite a draft to match the Global StyleProfile and stream the result.

    Requires that the global style has been trained offline.
    """
    mode = request.style_mode or "populer"
    profile_path = settings.profiles_path / f"{mode}_style.json"
    
    if not profile_path.exists():
        profile_path = settings.profiles_path / "global_style.json"
        
    if not profile_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Style profile not found. Run training script first.",
        )

    # Load the style profile
    try:
        profile_data = load_json(profile_path)
        profile_data["style_mode"] = request.style_mode
        style = StyleProfile(**profile_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load global style profile: {e}",
        )

    return StreamingResponse(
        apply_style_stream(request.draft, style), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


class EnhanceRequest(BaseModel):
    record_id:       int | None = None
    original_text:   str
    standard_output: str
    style_mode:      str = "populer"
    language:        str = "id"


class EnhancedResult(BaseModel):
    enhanced_text:     str
    judge_score_before: float   # skor Standard
    judge_score_after:  float   # skor Enhanced
    worst_sentence:    str
    improved:          bool     # True jika enhanced != standard
    trigram_overlap:   float


@router.post("/process/enhance", response_model=EnhancedResult)
async def enhance_output(request: EnhanceRequest):
    """
    Tahap 2: Terima standard output, jalankan LLM Judge,
    lalu Pass 3 jika judge_score < 65.
    Dipanggil setelah /process selesai streaming.
    """
    from app.core.evaluator import run_llm_judge, SQLiteEvaluator
    from app.core.writing_engine import (
        _apply_post_processing,
        _programmatic_sentence_humanize,
        _validate_paragraph_count,
        check_trigram_overlap,
        _clean_input_draft,
        _count_paragraphs,
        _build_system_prompt,
        _strip_meta_commentary,
    )
    from app.storage.json_store import load_style_profile
    from pydantic_ai import Agent
    from pydantic_ai.models.fallback import FallbackModel
    from pathlib import Path

    clean_original = _clean_input_draft(request.original_text)
    paragraph_count = _count_paragraphs(clean_original)

    # Step 1: Judge Standard output
    judge_result = await run_llm_judge(
        original_text=clean_original,
        humanized_text=request.standard_output,
        style_mode=request.style_mode,
        language=request.language,
    )

    score_before   = judge_result.get("overall_score", 100)
    worst_sentence = judge_result.get("worst_sentence", "")

    # Step 2: Jika score < 65 → Pass 3 targeted fix
    if score_before < 65 and worst_sentence:
        # Panggil get_global_style secara manual untuk load config default style_mode
        mode = request.style_mode or "populer"
        profile_path = settings.profiles_path / f"{mode}_style.json"
        if not profile_path.exists():
            profile_path = settings.profiles_path / "global_style.json"
            
        profile_data = load_json(profile_path)
        profile_data["style_mode"] = request.style_mode
        style = StyleProfile(**profile_data)
        
        system_prompt = _build_system_prompt(style, paragraph_count, request.language)

        agent = Agent(
            model=FallbackModel(
                "groq:llama-3.3-70b-versatile",
                "groq:llama-3.1-8b-instant",
            ),
            system_prompt=system_prompt,
        )

        pass3_msg = (
            f"TUGAS: Tulis ulang teks berikut.\n"
            f"ATURAN MUTLAK:\n"
            f"- Output HANYA teks yang sudah diubah\n"
            f"- DILARANG menulis penjelasan, catatan, "
            f"  atau daftar perubahan apapun\n"
            f"- DILARANG menulis kalimat seperti "
            f"  'berikut teks ulang' atau 'perubahan yang dilakukan'\n"
            f"- Langsung mulai dengan kalimat pertama teks\n\n"
            f"Teks berikut mendapat skor {score_before}/100 "
            f"dari evaluator karena kalimat ini masih terasa AI:\n"
            f"\"{worst_sentence}\"\n\n"
            f"Tulis ulang SELURUH teks — fokus perbaiki kalimat "
            f"di atas agar lebih natural. Pertahankan semua fakta "
            f"dan jumlah paragraf ({paragraph_count} paragraf).\n\n"
            f"{request.standard_output}"
        )

        try:
            result3 = await agent.run(
                pass3_msg,
                model_settings={"temperature": 1.0},
            )
            enhanced_raw = str(result3.output).strip()
            enhanced_raw = _strip_meta_commentary(enhanced_raw)
        except Exception:
            enhanced_raw = request.standard_output

        # Post-processing Pass 3
        enhanced = _apply_post_processing(
            enhanced_raw, request.language, request.style_mode
        )
        enhanced = _programmatic_sentence_humanize(
            enhanced, request.language, request.style_mode
        )
        enhanced = _validate_paragraph_count(
            enhanced, paragraph_count, clean_original
        )

        # Judge Enhanced
        judge_after = await run_llm_judge(
            original_text=clean_original,
            humanized_text=enhanced,
            style_mode=request.style_mode,
            language=request.language,
        )
        score_after = judge_after.get("overall_score", score_before)
        improved    = score_after > score_before

        # Gunakan enhanced hanya jika benar-benar lebih baik
        final_enhanced = enhanced if improved else request.standard_output
        final_score    = score_after if improved else score_before

    else:
        # Score sudah bagus — Enhanced = Standard
        final_enhanced = request.standard_output
        final_score    = score_before
        improved       = False

    trigram = check_trigram_overlap(clean_original, final_enhanced)

    # Update DB jika record_id ada
    if request.record_id:
        evaluator = SQLiteEvaluator(db_path=str(Path("data") / "evaluations.db"))
        evaluator.update_enhanced_result(
            record_id=request.record_id,
            enhanced_text=final_enhanced,
            enhanced_judge_score=final_score,
            enhanced_trigram=trigram,
            judge_score_standard=score_before
        )

    return EnhancedResult(
        enhanced_text=final_enhanced,
        judge_score_before=score_before,
        judge_score_after=final_score,
        worst_sentence=worst_sentence,
        improved=improved,
        trigram_overlap=trigram,
    )


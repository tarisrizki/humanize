import asyncio
import json
import re

from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from langdetect import detect
from typing import AsyncGenerator

from app.models.schemas import ProcessedText
from app.models.style_profile import StyleProfile
from app.core.prompts import _build_system_prompt
from app.core.text_utils import (
    _count_paragraphs, _clean_input_draft, _strip_meta_commentary, 
    check_trigram_overlap, _validate_paragraph_count, _enforce_min_sentences,
    _generate_changes_made, _validate_output_quality, _score_human_likelihood,
    _programmatic_sentence_humanize, _inject_short_sentences, _apply_post_processing
)

async def apply_style_stream(
    draft: str, style: StyleProfile
) -> AsyncGenerator[str, None]:
    clean_draft = _clean_input_draft(draft)
    paragraph_count = _count_paragraphs(clean_draft)
    
    try:
        input_lang = "id" if detect(clean_draft[:2000]) in ("id", "ms") else "en"
    except Exception:
        input_lang = "id"
        
    system_prompt = _build_system_prompt(style, paragraph_count)
    style_mode = getattr(style, 'style_mode', 'populer')

    drafter_agent = Agent(
        model=FallbackModel(
            "groq:llama-3.1-8b-instant",
            "groq:llama-3.3-70b-versatile",
        ),
        system_prompt=system_prompt,
    )

    user_msg = (
        f"Tulis ulang draf berikut agar terdengar natural, "
        f"ditulis oleh manusia sungguhan. "
        f"PENTING: setiap paragraf HARUS minimal 4 kalimat "
        f"dengan variasi panjang yang ekstrem. "
        f"Ikuti register dan gaya yang telah ditentukan. "
        f"Kembalikan HANYA teks hasil rewrite — "
        f"jangan tambahkan penjelasan, label, atau komentar (kecuali tag <thought> di awal). "
        f"Output WAJIB persis {paragraph_count} paragraf "
        f"dipisahkan baris kosong.\n\n"
        f"{clean_draft}"
    )

    try:
        result = await asyncio.wait_for(
            drafter_agent.run(
                user_msg,
                model_settings={"temperature": 1.1},
            ),
            timeout=60.0
        )
        full_text = str(result.output).strip() if result.output else ""
    except Exception as e:
        import logging
        logging.exception("Drafter agent failed (Timeout/API Error)")
        full_text = ""

    # Parse out <thought>
    import re as re_mod
    thought_match = re_mod.search(r'<thought>.*?</thought>', full_text, flags=re_mod.DOTALL | re_mod.IGNORECASE)
    thought_block = ""
    if thought_match:
        thought_block = thought_match.group(0)
        full_text = full_text.replace(thought_block, "").strip()

    full_text = _strip_meta_commentary(full_text)
    trigram_overlap = check_trigram_overlap(clean_draft, full_text)
    
    # Send thought block to frontend quickly if exists
    if thought_block:
        yield f"event: text\ndata: {json.dumps(thought_block + '\n\n')}\n\n"
        await asyncio.sleep(0.1)

    # Pass 2: Structural Rewrite if Turnitin overlap > 30%
    if trigram_overlap > 0.30:
        yield f"event: text\ndata: {json.dumps('<!-- turnitin_refine -->\n')}\n\n"
        pass2_msg = (
            f"Teks ini masih terlalu mirip strukturnya dengan draf asli (Overlap {trigram_overlap*100:.0f}% > 30%).\n"
            f"LAKUKAN STRUCTURAL REWRITE TOTAL: ubah urutan paragraf, gabungkan kalimat, "
            f"atau pecah paragraf. Jangan hanya mengganti sinonim.\n"
            f"Tulis perencanaanmu di dalam <thought> lalu hasil teks.\n"
            f"Berikut adalah teks yang perlu dirombak total:\n\n{full_text}"
        )
        try:
            result_pass2 = await asyncio.wait_for(
                drafter_agent.run(
                    pass2_msg,
                    model_settings={"temperature": 1.2},
                ),
                timeout=60.0
            )
            pass2_text = str(result_pass2.output).strip() if result_pass2.output else ""
            thought_match2 = re_mod.search(r'<thought>.*?</thought>', pass2_text, flags=re_mod.DOTALL | re_mod.IGNORECASE)
            if thought_match2:
                thought_block2 = thought_match2.group(0)
                yield f"event: text\ndata: {json.dumps(thought_block2 + '\n\n')}\n\n"
                pass2_text = pass2_text.replace(thought_block2, "").strip()
            full_text = _strip_meta_commentary(pass2_text)
            trigram_overlap = check_trigram_overlap(clean_draft, full_text)
        except Exception as e:
            import logging
            logging.exception("Pass 2 structural rewrite failed (Timeout/API Error)")

    # --- Apply Programmatic Humanization ---
    full_text = _programmatic_sentence_humanize(full_text, input_lang, style_mode)
    full_text = _apply_post_processing(full_text, input_lang, style_mode)
    full_text = _inject_short_sentences(full_text, input_lang, style_mode)

    # Pass 3: Editor Agent for Polishing (Streaming)
    editor_agent = Agent(
        model="groq:llama-3.1-8b-instant",
        system_prompt=(
            "Tugasmu HANYA SATU: membersihkan teks dari sisa-sisa gaya AI dan memastikan "
            "tata bahasanya luwes, natural, dan mengalir seperti tulisan jurnalis/penulis manusia senior. "
            "DILARANG KERAS mengubah makna, membuang kalimat, "
            "atau merusak jumlah paragraf. Panjang teks (jumlah kata) harus TETAP SAMA atau lebih panjang. "
            "Hapus transisi kaku seperti 'Oleh karena itu', "
            "'Selain itu', 'Hal ini menunjukkan', 'Dengan demikian', dll. Ganti dengan kata santai "
            "atau hilangkan sama sekali tanpa membuang informasi intinya.\n"
            "Langsung berikan teks akhir tanpa <thought> dan tanpa komentar."
        )
    )

    try:
        async with editor_agent.run_stream(
            full_text,
            model_settings={"temperature": 0.5},
        ) as editor_stream:
            final_polished_text = ""
            async for chunk in editor_stream.stream_text(delta=True):
                yield f"event: text\ndata: {json.dumps(chunk)}\n\n"
                final_polished_text += chunk
            full_text = final_polished_text
    except Exception as e:
        import logging
        logging.exception("Editor Agent stream failed (Timeout/API Error). Fallback to sync yield.")
        # Fallback to yielding synchronously if stream fails
        chunk_size = 10
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i+chunk_size]
            yield f"event: text\ndata: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)
        
    full_text = _validate_paragraph_count(full_text, paragraph_count, clean_draft)
    full_text = _enforce_min_sentences(full_text, min_sentences=4)

    final_trigram_overlap = check_trigram_overlap(clean_draft, full_text)
    changes = _generate_changes_made(clean_draft, full_text, final_trigram_overlap)
    yield f"event: metrics\ndata: {json.dumps({'changes_made': changes})}\n\n"


async def apply_style(
    draft: str, style: StyleProfile
) -> ProcessedText:
    stream_gen = apply_style_stream(draft, style)
    
    final_text = ""
    changes = []
    
    try:
        async for chunk in stream_gen:
            for line in chunk.split('\n'):
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str:
                        try:
                            data = json.loads(data_str)
                            if isinstance(data, dict) and "changes_made" in data:
                                changes = data["changes_made"]
                            elif isinstance(data, str) and "<!-- turnitin_refine -->" not in data and "<thought>" not in data:
                                final_text += data
                        except:
                            pass
    except Exception as e:
        import logging
        logging.exception("Error in apply_style")
    
    return ProcessedText(final_text=final_text.strip(), changes_made=changes)

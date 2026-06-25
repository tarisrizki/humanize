"""Writing Engine — Rewrites draft text to match a user's StyleProfile.

Uses PydanticAI with Groq to perform intelligent rewriting that evades AI detection.
"""

import asyncio
import json
import os
import re
import random
from typing import AsyncGenerator

from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel

from app.config import settings
from app.models.schemas import ProcessedText
from app.models.style_profile import StyleProfile


def _count_paragraphs(text: str) -> int:
    """Count paragraphs in text (separated by double newlines or single newlines with content)."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n', text) if p.strip()]
    return len(paragraphs)


def _clean_input_draft(draft: str) -> str:
    """Remove web artifacts from the input draft before sending to LLM."""
    patterns = [
        r'Enter fullscreen mode\.?',
        r'Exit fullscreen mode\.?',
        r'─+\s*─*',
        r'Click here.*',
        r'Read more.*',
    ]
    for p in patterns:
        draft = re.sub(p, '', draft, flags=re.IGNORECASE)
    draft = re.sub(r'\n{3,}', '\n\n', draft)
    draft = re.sub(r' {2,}', ' ', draft)
    return draft.strip()


def _build_system_prompt(style: StyleProfile, paragraph_count: int) -> str:
    """Generate a detailed system prompt heavily weighting few-shot examples."""

    lang_instruction = ""
    if style.language == "id":
        lang_instruction = """
LANGUAGE: The user writes in Indonesian (Bahasa Indonesia).
- Maintain natural Indonesian phrasing and grammar.
- Use appropriate Indonesian formal/informal register as indicated.
- Do NOT translate to English.
"""
    elif style.language == "mixed":
        lang_instruction = """
LANGUAGE: The user writes in a mix of Indonesian and English.
- Maintain the same code-switching pattern.
- Keep the same ratio of Indonesian to English.
"""
    else:
        lang_instruction = """
LANGUAGE: The user writes in English.
- Maintain natural English phrasing and grammar.
"""

    few_shot_text = ""
    if style.few_shot_examples:
        examples_str = "\n\n".join([f"Example {i+1}:\n{ex}" for i, ex in enumerate(style.few_shot_examples)])
        few_shot_text = f"""
## PRIMARY DIRECTIVE: FEW-SHOT EXAMPLES
The following paragraphs are examples of the author's ACTUAL writing. 
You MUST use these examples as your primary stylistic ground truth. Match the cadence, phrasing, and natural flow of these examples exactly.

{examples_str}
"""

    return f"""You are a native Indonesian human writer, NOT an AI assistant. You are rewriting a draft 
to sound like it was written by a real person with a unique voice. Your goal is to make the text 
COMPLETELY UNDETECTABLE by AI detection tools like GPTZero, Originality.ai, and ZeroGPT.

{lang_instruction}

{few_shot_text}

## STYLE METRICS
- Average sentence length: ~{style.avg_sentence_length} words
- Active voice ratio: {style.active_voice_ratio:.0%}
- Preferred transitions: {", ".join(style.top_transitions) if style.top_transitions else "minimal"}
- Tone & Register: {style.emotion_register}
- Favorite phrases: {", ".join(f'"{p}"' for p in style.favorite_phrases) if style.favorite_phrases else "none identified"}

## CRITICAL ANTI-DETECTION RULES

1. PRESERVE all factual content and meaning from the original draft.
2. DO NOT add new information that wasn't in the draft.

3. EXTREME BURSTINESS (MOST IMPORTANT):
   - You MUST alternate between very short sentences (2-5 words) and long complex sentences (25-40 words).
   - Every paragraph MUST contain at least one sentence of 4 words or fewer.
   - Example rhythm: "Banjir datang tiba-tiba. [short] Warga yang tinggal di bantaran sungai langsung berhamburan keluar rumah sambil membawa barang-barang berharga mereka yang sempat diselamatkan. [very long] Panik. [ultra short] Beberapa orang tua kesulitan berjalan karena air sudah setinggi lutut dan arus cukup deras. [long]"

4. UNPREDICTABLE VOCABULARY:
   - NEVER use these AI-typical words/phrases: "merupakan", "memiliki", "berbagai", "melalui", "serta", 
     "selain itu", "oleh karena itu", "secara keseluruhan", "penting untuk", "di sisi lain", 
     "perlu dicatat", "hal ini", "dalam hal ini", "dapat disimpulkan", "sangat penting"
   - Use informal alternatives: "merupakan" → "itu", "memiliki" → "punya", "berbagai" → "macam-macam", 
     "melalui" → "lewat", "serta" → "dan juga"

5. HUMAN IMPERFECTIONS:
   - Start some sentences with conjunctions: "Dan...", "Tapi...", "Soalnya..."
   - Use rhetorical questions: "Apa yang terjadi selanjutnya?", "Siapa yang mau disalahkan?"
   - Add parenthetical asides: "(ya, memang begitu kenyataannya)"
   - Occasionally use first person: "kalau saya lihat...", "menurut saya..."
   - Use colloquial connectors: "nah", "jadi begini", "yang jelas"

6. PARAGRAPH STRUCTURE:
   - The original draft has exactly {paragraph_count} paragraph(s).
   - Your rewrite MUST have EXACTLY {paragraph_count} paragraph(s), separated by blank lines.
   - DO NOT merge or split paragraphs.

7. NO AI PATTERNS:
   - Never start consecutive sentences with the same word.
   - Never use parallel sentence structures (e.g., "X menyebabkan Y. A menyebabkan B.")
   - Don't end with a neat summary or conclusion unless the original does.
   - Don't use overly formal or academic tone.

8. List all significant changes you made in changes_made.
"""


def _score_sentence_ai_likeness(sentence: str, lang: str = "id") -> bool:
    """Return True if sentence looks AI-generated."""
    s = sentence.lower().strip()
    score = 0
    
    if lang in ("id", "mixed"):
        ai_starters = [
            "hal ini", "selain itu", "dengan demikian", "dapat ",
            "perlu ", "dalam hal", "sebagai ", "oleh karena", 
            "secara ", "berdasarkan", "di samping", "sementara itu",
            "adapun", "terkait", "melalui", "untuk mencapai",
        ]
        ai_words = [
            "merupakan", "memiliki", "berbagai", "sehingga",
            "serta", "tersebut", "dilakukan", "digunakan",
            "menunjukkan bahwa", "dapat disimpulkan", "penting untuk",
        ]
    else:
        ai_starters = [
            "this ", "it is", "in addition", "furthermore",
            "moreover", "additionally", "consequently", "therefore",
            "in conclusion", "it should", "in order to", "as a result",
        ]
        ai_words = [
            "utilize", "leverage", "facilitate", "demonstrate",
            "indicate", "significant", "crucial", "ensure", "implement",
        ]
    
    # AI starter: +3
    for starter in ai_starters:
        if s.startswith(starter):
            score += 3
            break
    
    # Uniform length (10-20 words): +1
    words = s.split()
    if 10 <= len(words) <= 20:
        score += 1
    
    # No personal voice: +1
    personal = ["saya", "kami", "kita", "gue", "aku"] if lang in ("id","mixed") \
               else ["i ", "we ", "my ", "our "]
    if not any(m in s for m in personal):
        score += 1
    
    # AI-typical words found: +2
    if any(w in s for w in ai_words):
        score += 2
    
    # No questions or exclamations: +0.5
    if "?" not in sentence and "!" not in sentence:
        score += 0.5
    
    return score >= 3.5


async def _rewrite_flagged_sentences(
    text: str, 
    lang: str,
    style_mode: str,
    agent,
    max_passes: int = 2,
) -> str:
    """Identify AI-flagged sentences and rewrite only those."""
    
    if style_mode == "akademik":
        persona = "akademisi yang menulis dengan gaya personal namun tetap ilmiah"
        avoid = "Hindari: 'hal ini', 'dapat disimpulkan', 'menunjukkan bahwa', 'tersebut'"
    elif style_mode == "profesional":
        persona = "profesional berpengalaman yang menulis ringkas dan to-the-point"
        avoid = "Hindari: 'hal ini', 'berbagai', 'sehingga', 'tersebut', kalimat pasif berlebihan"
    elif style_mode == "kreatif":
        persona = "penulis kreatif dengan gaya naratif yang hidup dan emosional"
        avoid = "Hindari: kalimat formal, struktur subjek-predikat-objek yang robotik"
    else:  # populer
        persona = "blogger santai yang nulis seperti ngobrol dengan teman"
        avoid = "Hindari: 'hal ini', 'berbagai', 'merupakan', 'tersebut', kalimat pasif"
    
    system_prompt = f"""Kamu {persona}. Tugas kamu: tulis ulang kalimat-kalimat yang terdengar 
seperti ditulis AI, agar terdengar natural dan manusiawi.

{avoid}

ATURAN:
- Ganti kalimat yang dimulai dengan kata AI-typical
- Pecah kalimat panjang seragam menjadi variasi pendek+panjang
- Tambahkan pertanyaan retoris atau ekspresi personal jika sesuai
- Pertahankan makna aslinya
- HANYA kembalikan kalimat yang sudah ditulis ulang, 
  dengan format yang sama persis (nomor. kalimat)"""
    
    from pydantic_ai import Agent as SimpleAgent
    from pydantic_ai.models.fallback import FallbackModel
    
    plain_agent = SimpleAgent(
        model=FallbackModel(
            "groq:llama-3.3-70b-versatile", 
            "groq:llama-3.1-8b-instant"
        ),
        system_prompt=system_prompt,
    )
    
    for pass_num in range(max_passes):
        # Split preserving paragraph structure
        paragraphs = text.split('\n')
        changed = False
        
        for p_idx, para in enumerate(paragraphs):
            if not para.strip():
                continue
            
            sentences = re.split(r'(?<=[.!?])\s+', para.strip())
            flagged = [
                (i, s) for i, s in enumerate(sentences)
                if _score_sentence_ai_likeness(s, lang) and len(s.split()) > 4
            ]
            
            if not flagged:
                continue
            
            # Build rewrite request for this paragraph's flagged sentences
            input_lines = "\n".join([
                f"{i+1}. {s}" for i, s in flagged
            ])
            
            try:
                result = await plain_agent.run(
                    f"Tulis ulang kalimat-kalimat ini:\n\n{input_lines}",
                    model_settings={"temperature": 1.4}
                )
                output = result.output if isinstance(result.output, str) else str(result.output)
                
                # Parse numbered output
                rewrites = {}
                for line in output.strip().split('\n'):
                    m = re.match(r'^(\d+)\.\s+(.+)$', line.strip())
                    if m:
                        rewrites[int(m.group(1)) - 1] = m.group(2).strip()
                
                # Apply rewrites
                for orig_idx, _ in flagged:
                    if orig_idx in rewrites and rewrites[orig_idx]:
                        sentences[orig_idx] = rewrites[orig_idx]
                        changed = True
                        
                paragraphs[p_idx] = ' '.join(sentences)
                
            except Exception:
                continue  # Skip if rewrite fails, keep original
        
        text = '\n'.join(paragraphs)
        
        if not changed:
            break
    
    return text


def _inject_short_sentences(text: str, lang: str, style_mode: str = "populer") -> str:
    if lang in ("id", "mixed"):
        short_injects = ["Parah.", "Serius.", "Pusing.", 
            "Gila sih.", "Capek banget.", "Aneh emang.", 
            "Nggak masuk akal.", "Ribet.", "Makanya."]
    else:
        short_injects = ["Seriously.", "Wild.", "For real.",
            "Crazy.", "No joke.", "Right?", "Exactly."]
    paragraphs = text.split('\n')
    result = []
    for para in paragraphs:
        if not para.strip():
            result.append(para)
            continue
        sentences = re.split(r'(?<=[.!?])\s+', para.strip())
        new_sentences = []
        for sent in sentences:
            new_sentences.append(sent)
            if len(sent.split()) > 15 and random.random() < 0.30:
                new_sentences.append(random.choice(short_injects))
        result.append(' '.join(new_sentences))
    return '\n'.join(result)


def _apply_post_processing(text: str, lang: str, style_mode: str = "populer") -> str:
    """Apply aggressive post-processing to break AI detection patterns."""
    if not text:
        return text
    
    # First, clean any web artifacts that might have leaked through
    web_artifacts = [
        r'Enter fullscreen mode\.?',
        r'Exit fullscreen mode\.?',
        r'─+\s*─*',
    ]
    for p in web_artifacts:
        text = re.sub(p, '', text, flags=re.IGNORECASE)
        
    if lang in ("id", "mixed"):
        # Replace AI-typical Indonesian words/phrases
        replacements = [
            (r"(?i)\bsecara keseluruhan\b", "singkat cerita"),
            (r"(?i)\bselain itu\b", "satu hal lagi"),
            (r"(?i)\bpenting untuk diingat\b", "jangan lupa"),
            (r"(?i)\bkesimpulannya\b", "intinya"),
            (r"(?i)\boleh karena itu\b", "makanya"),
            (r"(?i)\bdi sisi lain\b", "tapi kalau dilihat dari sisi lain"),
            (r"(?i)\bperlu dicatat\b", "yang perlu digarisbawahi"),
            (r"(?i)\bdapat disimpulkan\b", "bisa dibilang"),
            (r"(?i)\bsangat penting\b", "krusial"),
            (r"(?i)\bmerupakan\b", "itu"),
            (r"(?i)\badalah\b", "itu"),
            (r"(?i)\bmemiliki\b", "punya"),
            (r"(?i)\bberbagai macam\b", "banyak"),
            (r"(?i)\bberbagai\b", "banyak"),
            (r"(?i)\bmelalui\b", "lewat"),
            (r"(?i)\bdalam hal ini\b", "soal ini"),
            (r"(?i)\bhal ini\b", "ini"),
            (r"(?i)\bnamun demikian\b", "tapi ya"),
            (r"(?i)\bdengan demikian\b", "jadi"),
            (r"(?i)\bberdasarkan\b", "menurut"),
            (r"(?i)\bsebagaimana\b", "seperti"),
            (r"(?i)\bterhadap\b", "pada"),
            (r"(?i)\bsehingga\b", "jadi"),
            (r"(?i)\bsekaligus\b", "dan juga"),
            (r"(?i)\bmeliputi\b", "termasuk"),
            (r"(?i)\bmenimbulkan\b", "bikin"),
        ]
        conversational_injects = ["Dan ", "Tapi ", "Nah, ", "Soalnya, ", "Yang jelas, "]
    else:
        replacements = [
            (r"(?i)\bin conclusion\b", "to wrap up"),
            (r"(?i)\bfurthermore\b", "what's more"),
            (r"(?i)\bit is important to note\b", "keep in mind"),
            (r"(?i)\bmoreover\b", "plus"),
            (r"(?i)\badditionally\b", "on top of that"),
            (r"(?i)\bconsequently\b", "so"),
            (r"(?i)\bnevertheless\b", "still"),
            (r"(?i)\bsubsequently\b", "then"),
            (r"(?i)\bit is worth noting\b", "worth mentioning"),
            (r"(?i)\bin order to\b", "to"),
        ]
        conversational_injects = ["And ", "But ", "Look, ", "Thing is, ", "Honestly, "]
        
    # Apply word replacements
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    
    # Process paragraph by paragraph to preserve structure
    paragraphs = text.split('\n')
    for p_idx, p_text in enumerate(paragraphs):
        if not p_text.strip():
            continue
            
        if lang in ("id", "mixed"):
            sentences = re.split(r'(?<=[.!?])\s+', p_text)
        else:
            sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', p_text)
            
        if len(sentences) > 1:
            for i in range(1, len(sentences)):
                if random.random() < 0.15:  # 15% chance
                    inj = random.choice(conversational_injects)
                    first_char = sentences[i][0].lower() if len(sentences[i]) > 0 else ""
                    sentences[i] = inj + first_char + sentences[i][1:]
            paragraphs[p_idx] = " ".join(sentences)
            
    text = "\n".join(paragraphs)
    
    # Clean up any double spaces or periods
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


# Set API key at module level
os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

async def apply_style_stream(draft: str, style: StyleProfile) -> AsyncGenerator[str, None]:
    clean_draft = _clean_input_draft(draft)
    paragraph_count = _count_paragraphs(clean_draft)
    system_prompt = _build_system_prompt(style, paragraph_count)
    
    agent = Agent(
        model=FallbackModel(
            "groq:llama-3.3-70b-versatile", 
            "groq:llama-3.1-8b-instant"
        ),
        system_prompt=system_prompt,
        output_type=ProcessedText,
    )
    
    style_mode = getattr(style, 'style_mode', 'populer')
    score_lang = "id" if style.language in ("id", "mixed") else "en"
    
    # ── PASS 1: Full rewrite ──────────────────────────────────
    user_msg = (
        f"Rewrite this draft to sound like a natural human wrote it. "
        f"Keep exactly {paragraph_count} paragraphs:\n\n{clean_draft}"
    )
    
    async with agent.run_stream(
        user_msg,
        model_settings={"temperature": 1.3}
    ) as result:
        full_text = ""
        final_result = None
        async for partial_msg in result.stream_output():
            final_result = partial_msg
            if partial_msg.final_text:
                full_text = partial_msg.final_text
    
    if not full_text or not final_result:
        return
    
    # ── POST-PROCESS ─────────────────────────────────────────
    text = _apply_post_processing(full_text, style.language, style_mode)
    text = _inject_short_sentences(text, style.language, style_mode)
    
    # ── PASS 2: Sentence-level targeted rewrite ───────────────
    text = await _rewrite_flagged_sentences(
        text, style.language, style_mode, agent
    )
    
    # ── Light post-process again ──────────────────────────────
    text = _apply_post_processing(text, style.language, style_mode)
    
    # ── Stream result ─────────────────────────────────────────
    chunk_size = 10
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        yield f"event: text\ndata: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)
    
    metrics = {
        "changes_made": final_result.changes_made or [],
    }
    yield f"event: metrics\ndata: {json.dumps(metrics)}\n\n"


async def apply_style(draft: str, style: StyleProfile) -> ProcessedText:
    clean_draft = _clean_input_draft(draft)
    paragraph_count = _count_paragraphs(clean_draft)
    system_prompt = _build_system_prompt(style, paragraph_count)
    style_mode = getattr(style, 'style_mode', 'populer')
    score_lang = "id" if style.language in ("id", "mixed") else "en"
    
    agent = Agent(
        model=FallbackModel(
            "groq:llama-3.3-70b-versatile",
            "groq:llama-3.1-8b-instant"
        ),
        system_prompt=system_prompt,
        output_type=ProcessedText,
    )
    
    user_msg = (
        f"Rewrite this draft to sound like a natural human wrote it. "
        f"Keep exactly {paragraph_count} paragraphs:\n\n{clean_draft}"
    )
    
    result = await agent.run(user_msg, model_settings={"temperature": 1.3})
    text = result.output.final_text
    
    text = _apply_post_processing(text, style.language, style_mode)
    text = _inject_short_sentences(text, style.language, style_mode)
    text = await _rewrite_flagged_sentences(
        text, style.language, style_mode, agent
    )
    text = _apply_post_processing(text, style.language, style_mode)
    
    result.output.final_text = text
    return result.output

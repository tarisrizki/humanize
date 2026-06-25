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
from app.core.detection_scorer import score_humanness
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


def _inject_short_sentences(text: str, lang: str) -> str:
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


def _apply_post_processing(text: str, lang: str) -> str:
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
    """Rewrite a draft to match the given StyleProfile, streaming the output via SSE.
    
    Yields Server-Sent Events formatted strings:
    event: text
    data: <chunk>
    
    event: metrics
    data: <json>
    """
    # Clean input and count paragraphs
    clean_draft = _clean_input_draft(draft)
    paragraph_count = _count_paragraphs(clean_draft)
    
    system_prompt = _build_system_prompt(style, paragraph_count)

    agent = Agent(
        model=FallbackModel("groq:llama-3.3-70b-versatile", "groq:llama-3.1-8b-instant"),
        system_prompt=system_prompt,
        output_type=ProcessedText,
    )
    
    max_attempts = 3
    current_draft = f"Rewrite this draft to sound like a natural human wrote it. The draft has {paragraph_count} paragraphs — keep exactly {paragraph_count} paragraphs in your output:\n\n{clean_draft}"
    
    for attempt in range(max_attempts):
        is_last_attempt = (attempt == max_attempts - 1)
        
        async with agent.run_stream(
            current_draft,
            model_settings={"temperature": 1.3 + (attempt * 0.1)}  # Increase temp each retry
        ) as result:
            # Buffer the entire response for post-processing
            full_text = ""
            final_result = None
            
            async for partial_msg in result.stream_output():
                final_result = partial_msg
                if partial_msg.final_text:
                    full_text = partial_msg.final_text
                    
            # Post-process the fully assembled text
            if final_result and full_text:
                processed_text = _apply_post_processing(full_text, style.language)
                processed_text = _inject_short_sentences(processed_text, style.language)
                score_lang = "id" if style.language in ("id", "mixed") else "en"
                score = score_humanness(processed_text, lang=score_lang)
                
                if score.get("is_human_like", True) or is_last_attempt:
                    final_result.final_text = processed_text
                    
                    # Simulate streaming the processed text
                    chunk_size = 10
                    for i in range(0, len(processed_text), chunk_size):
                        chunk = processed_text[i:i+chunk_size]
                        safe_chunk = json.dumps(chunk)
                        yield f"event: text\ndata: {safe_chunk}\n\n"
                        await asyncio.sleep(0.01)
                        
                    metrics = {
                        "changes_made": final_result.changes_made or [],
                        "humanness_score": score
                    }
                    yield f"event: metrics\ndata: {json.dumps(metrics)}\n\n"
                    break
                else:
                    # Failed humanness check, rewrite with more aggressive instruction
                    current_draft = f"""The previous rewrite FAILED the human detection test (burstiness too low: {score.get('burstiness_score', 0)}).

CRITICAL INSTRUCTION: You MUST use EXTREME sentence length variation:
- Include at least 3 sentences of 4 words or fewer (e.g., "Banjir melanda.", "Parah sekali.", "Tidak ada solusi.")
- Include at least 2 sentences of 30+ words
- Start some sentences with "Dan", "Tapi", "Nah,"
- Use rhetorical questions

Rewrite this text with {paragraph_count} paragraphs:

{processed_text}"""

async def apply_style(draft: str, style: StyleProfile) -> ProcessedText:
    """Rewrite a draft to match the given StyleProfile.

    Args:
        draft: The original draft text to rewrite.
        style: The target writing style to match.

    Returns:
        ProcessedText with final_text and changes_made.
    """
    # Clean input and count paragraphs
    clean_draft = _clean_input_draft(draft)
    paragraph_count = _count_paragraphs(clean_draft)
    
    system_prompt = _build_system_prompt(style, paragraph_count)

    agent = Agent(
        model=FallbackModel("groq:llama-3.3-70b-versatile", "groq:llama-3.1-8b-instant"),
        system_prompt=system_prompt,
        output_type=ProcessedText,
    )

    max_attempts = 3
    current_draft = f"Rewrite this draft to sound like a natural human wrote it. The draft has {paragraph_count} paragraphs — keep exactly {paragraph_count} paragraphs in your output:\n\n{clean_draft}"
    
    for attempt in range(max_attempts):
        is_last_attempt = (attempt == max_attempts - 1)
        
        result = await agent.run(
            current_draft,
            model_settings={"temperature": 1.3 + (attempt * 0.1)}
        )

        final_text = _apply_post_processing(result.output.final_text, style.language)
        final_text = _inject_short_sentences(final_text, style.language)
        result.output.final_text = final_text
        
        score_lang = "id" if style.language in ("id", "mixed") else "en"
        score = score_humanness(final_text, lang=score_lang)
        if score.get("is_human_like", True) or is_last_attempt:
            return result.output
        else:
            current_draft = f"""The previous rewrite FAILED the human detection test (burstiness too low: {score.get('burstiness_score', 0)}).

CRITICAL INSTRUCTION: You MUST use EXTREME sentence length variation:
- Include at least 3 sentences of 4 words or fewer
- Include at least 2 sentences of 30+ words
- Start some sentences with "Dan", "Tapi", "Nah,"
- Use rhetorical questions

Rewrite this text with {paragraph_count} paragraphs:

{final_text}"""
            
    return result.output

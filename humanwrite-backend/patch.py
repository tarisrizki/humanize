import re

with open('app/core/writing_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove import score_humanness
content = re.sub(
    r'from app\.core\.detection_scorer import score_humanness\n',
    '',
    content
)

# 2. Add style_mode to _inject_short_sentences
content = content.replace(
    'def _inject_short_sentences(text: str, lang: str) -> str:',
    'def _inject_short_sentences(text: str, lang: str, style_mode: str = "populer") -> str:'
)

# 3. Add style_mode to _apply_post_processing
content = content.replace(
    'def _apply_post_processing(text: str, lang: str) -> str:',
    'def _apply_post_processing(text: str, lang: str, style_mode: str = "populer") -> str:'
)

# 4. Insert _score_sentence_ai_likeness and _rewrite_flagged_sentences before _inject_short_sentences
insert_code = '''def _score_sentence_ai_likeness(sentence: str, lang: str = "id") -> bool:
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
    personal = ["saya", "kami", "kita", "gue", "aku"] if lang in ("id","mixed") \\
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
        paragraphs = text.split('\\n')
        changed = False
        
        for p_idx, para in enumerate(paragraphs):
            if not para.strip():
                continue
            
            sentences = re.split(r'(?<=[.!?])\\s+', para.strip())
            flagged = [
                (i, s) for i, s in enumerate(sentences)
                if _score_sentence_ai_likeness(s, lang) and len(s.split()) > 4
            ]
            
            if not flagged:
                continue
            
            # Build rewrite request for this paragraph's flagged sentences
            input_lines = "\\n".join([
                f"{i+1}. {s}" for i, s in flagged
            ])
            
            try:
                result = await plain_agent.run(
                    f"Tulis ulang kalimat-kalimat ini:\\n\\n{input_lines}",
                    model_settings={"temperature": 1.4}
                )
                output = result.output if isinstance(result.output, str) else str(result.output)
                
                # Parse numbered output
                rewrites = {}
                for line in output.strip().split('\\n'):
                    m = re.match(r'^(\\d+)\\.\\s+(.+)$', line.strip())
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
        
        text = '\\n'.join(paragraphs)
        
        if not changed:
            break
    
    return text


def _inject_short_sentences'''

content = content.replace('def _inject_short_sentences', insert_code)

# 5. Replace apply_style_stream and apply_style
new_apply = '''async def apply_style_stream(draft: str, style: StyleProfile) -> AsyncGenerator[str, None]:
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
        f"Keep exactly {paragraph_count} paragraphs:\\n\\n{clean_draft}"
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
        yield f"event: text\\ndata: {json.dumps(chunk)}\\n\\n"
        await asyncio.sleep(0.01)
    
    metrics = {
        "changes_made": final_result.changes_made or [],
    }
    yield f"event: metrics\\ndata: {json.dumps(metrics)}\\n\\n"


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
        f"Keep exactly {paragraph_count} paragraphs:\\n\\n{clean_draft}"
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
'''

# Find the start of apply_style_stream
stream_idx = content.find('async def apply_style_stream')
if stream_idx != -1:
    content = content[:stream_idx] + new_apply

with open('app/core/writing_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done patching')

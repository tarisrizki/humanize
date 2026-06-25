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
    style_mode = getattr(style, 'style_mode', 'populer')

    lang_instruction = ""
    if style.language == "id":
        lang_instruction = "Tulis dalam Bahasa Indonesia yang baik dan benar."
    elif style.language == "mixed":
        lang_instruction = "Pertahankan pola campur kode (Indonesia-Inggris) yang sama."
    else:
        lang_instruction = "Write in natural English."

    few_shot_text = ""
    if style.few_shot_examples:
        examples_str = "\n\n".join([
            f"Contoh {i+1}:\n{ex}" 
            for i, ex in enumerate(style.few_shot_examples)
        ])
        few_shot_text = f"""
## PANDUAN GAYA (WAJIB DIIKUTI)
Berikut contoh nyata tulisan yang harus dijadikan acuan gaya.
Tiru irama kalimat, pilihan kata, dan alur narasinya.

{examples_str}
"""

    if style_mode == "akademik":
        register_rules = """
## REGISTER: AKADEMIK
- Gunakan bahasa formal baku sesuai EYD.
- Kalimat pasif DIPERBOLEHKAN (lazim dalam tulisan ilmiah).
- Gunakan hedging: "tampaknya", "cenderung", "mengindikasikan bahwa".
- HINDARI frasa AI generik: "dapat disimpulkan bahwa", 
  "secara keseluruhan", "hal ini menunjukkan bahwa", "sangat penting".
- Kata "merupakan", "adalah", "berdasarkan", "sehingga" BOLEH dipakai.
- Variasikan panjang kalimat: mix kalimat analitik panjang (20-30 kata)
  dengan observasi singkat (5-10 kata) — ini penting.
- Contoh variasi: "Fenomena ini mencerminkan tren global. 
  Menarik, bukan?" atau "Data menunjukkan korelasi positif. Signifikan."
- BURSTINESS WAJIB: setiap paragraf harus punya
  minimal 1 kalimat sangat pendek (3-6 kata) dan
  1 kalimat panjang (20+ kata). Contoh:
  "Temuannya signifikan. [pendek] Analisis mendalam 
  terhadap data longitudinal selama lima tahun 
  menunjukkan korelasi yang konsisten antara variabel 
  independen dan hasil yang diamati. [panjang] 
  Patut dicermati. [pendek]"
"""
    elif style_mode == "profesional":
        register_rules = """
## REGISTER: PROFESIONAL
- Gunakan bahasa formal-fungsional sesuai EYD.
- Kalimat aktif, ringkas, dan to-the-point.
- HINDARI: "hal ini", "tersebut", "sangat penting", 
  "dapat disimpulkan", "secara keseluruhan".
- Variasikan panjang kalimat: mix kalimat panjang yang informatif 
  dengan pernyataan pendek yang tegas (3-6 kata).
- Contoh: "Hasilnya signifikan. Namun perlu analisis lebih lanjut 
  sebelum keputusan diambil."
- BURSTINESS WAJIB: mix kalimat tegas pendek (3-6 kata)
  dengan kalimat informatif panjang (18+ kata).
  Jangan semua kalimat punya panjang seragam.
"""
    elif style_mode == "kreatif":
        register_rules = """
## REGISTER: KREATIF  
- Utamakan ritme dan emosi, bukan struktur formal.
- Gunakan diksi yang vivid, metafora, dan detail sensoris.
- Variasi panjang kalimat EKSTREM: dari 2-3 kata hingga 30-35 kata.
- Boleh mulai kalimat dengan "Dan", "Tapi", "Karena", "Meski".
- HINDARI: penjelasan terlalu eksplisit, frasa klise AI.
- Show, don't tell.
"""
    else:  # populer
        register_rules = """
## REGISTER: POPULER
- Gunakan bahasa yang mudah dipahami, mengalir natural.
- Tetap mengikuti EYD — bukan bahasa gaul atau slang.
- Boleh conversational tapi pilihan kata tetap baku.
- HINDARI: "hal ini", "tersebut", "dapat disimpulkan", 
  "secara keseluruhan", "sangat penting", "merupakan".
- Variasikan panjang kalimat: mix kalimat panjang yang mengalir
  dengan kalimat pendek yang tegas.
- Contoh: "Situasinya memang rumit. Tapi bukan berarti tidak bisa 
  diselesaikan — asalkan ada kemauan dari semua pihak."
- BURSTINESS WAJIB: setiap paragraf harus mengandung
  kalimat sangat pendek (3-5 kata) dan kalimat 
  panjang yang mengalir (20+ kata). Variasi ini 
  membuat tulisan terasa hidup dan natural.
"""

    return f"""Kamu penulis manusia, bukan AI. Tugas kamu: tulis ulang draf ini
agar terdengar benar-benar ditulis manusia dan tidak terdeteksi 
oleh detektor AI seperti GPTZero.

{lang_instruction}

{few_shot_text}

{register_rules}

## ATURAN UNIVERSAL ANTI-DETEKSI
1. PERTAHANKAN semua fakta dan makna dari draf asli.
2. JANGAN tambahkan informasi baru.
3. BURSTINESS EKSTREM: setiap paragraf WAJIB mengandung
   minimal satu kalimat pendek (≤5 kata) DAN satu kalimat
   panjang (≥20 kata). Variasi ini krusial.
4. JANGAN mulai dua kalimat berturut dengan kata yang sama.
5. JANGAN gunakan struktur paralel berulang.
6. Tambahkan pertanyaan retoris atau observasi personal 
   yang sesuai register.
7. Output HARUS persis {paragraph_count} paragraf.
8. Catat semua perubahan signifikan di changes_made.
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


def _programmatic_sentence_humanize(text: str, lang: str, style_mode: str = "populer") -> str:
    """Rule-based sentence transformation. No LLM — 100% reliable."""
    if lang not in ("id", "mixed"):
        return text

    if style_mode in ("akademik", "profesional"):
        opener_rules = [
            # Formal: ganti AI-opener tapi tetap baku
            (r'^Hal ini menunjukkan bahwa ', 'Temuan ini mengindikasikan bahwa '),
            (r'^Hal ini membuktikan bahwa ', 'Data ini membuktikan bahwa '),
            (r'^Hal ini ', 'Kondisi ini '),
            (r'^Dapat disimpulkan bahwa ', 'Dapat dikatakan bahwa '),
            (r'^Dapat diketahui bahwa ', 'Terlihat bahwa '),
            (r'^Selain itu, ', 'Di samping itu, '),
            (r'^Selain itu ', 'Di samping itu '),
            (r'^Dengan demikian, ', 'Dengan begitu, '),
            (r'^Oleh karena itu, ', 'Karena itu, '),
            (r'^Berdasarkan hal tersebut', 'Berdasarkan hal di atas'),
            (r'^Secara keseluruhan, ', 'Secara umum, '),
            (r'^Sementara itu, ', 'Di sisi lain, '),
            (r'^Adapun ', 'Adapun '),
            # Jangan ganti: Berdasarkan, Melalui, Terhadap — kata baku
        ]
    elif style_mode == "kreatif":
        opener_rules = [
            (r'^Hal ini menunjukkan bahwa ', 'Ternyata, '),
            (r'^Hal ini ', 'Ini '),
            (r'^Dapat disimpulkan bahwa ', 'Pada akhirnya, '),
            (r'^Selain itu, ', 'Dan '),
            (r'^Dengan demikian, ', 'Maka, '),
            (r'^Oleh karena itu, ', 'Karena itu, '),
            (r'^Sementara itu, ', 'Sementara '),
            (r'^Adapun ', ''),
        ]
    else:  # populer
        opener_rules = [
            # Conversational tapi tetap EYD
            (r'^Hal ini menunjukkan bahwa ', 'Ternyata, '),
            (r'^Hal ini membuktikan bahwa ', 'Buktinya, '),
            (r'^Hal ini ', 'Ini '),
            (r'^Dapat disimpulkan bahwa ', 'Intinya, '),
            (r'^Dapat diketahui bahwa ', 'Yang jelas, '),
            (r'^Selain itu, ', 'Selain itu juga, '),
            (r'^Selain itu ', 'Dan '),
            (r'^Dengan demikian, ', 'Dengan begitu, '),
            (r'^Oleh karena itu, ', 'Karena itu, '),
            (r'^Berdasarkan hal tersebut', 'Dari situ'),
            (r'^Berdasarkan ', 'Menurut '),
            (r'^Secara keseluruhan, ', 'Singkat kata, '),
            (r'^Sementara itu, ', 'Di sisi lain, '),
            (r'^Adapun ', ''),
        ]

    # Word-level replacements (applied after opener rules)
    word_rules = [
        # "tersebut" sering bisa diganti "ini/itu" tanpa merusak EYD
        (r'\btersebut\b', 'ini'),
        # Frasa AI yang bisa diganti tanpa merusak struktur
        (r'\bmenunjukkan bahwa\b', 'mengindikasikan bahwa'),
        (r'\bdapat disimpulkan\b', 'dapat dikatakan'),
        # HAPUS: sangat penting → krusial (krusial juga terasa AI)
        # Ganti dengan yang lebih natural:
        (r'\bsangat penting untuk\b', 'penting untuk'),
    ]

    paragraphs = text.split('\n')
    result_paragraphs = []

    for para in paragraphs:
        if not para.strip():
            result_paragraphs.append(para)
            continue

        sentences = re.split(r'(?<=[.!?])\s+', para.strip())
        new_sentences = []

        for sent in sentences:
            s = sent.strip()
            if not s:
                continue

            # Apply opener rule (first match wins)
            for pattern, replacement in opener_rules:
                new_s = re.sub(pattern, replacement, s, count=1, flags=re.IGNORECASE)
                if new_s != s:
                    # Re-capitalize
                    s = new_s[0].upper() + new_s[1:] if new_s else new_s
                    break

            # Apply word rules
            for pattern, replacement in word_rules:
                s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)

            # Split very long sentence at "sehingga"
            if len(s.split()) > 22:
                parts = re.split(r'\bsehingga\b', s, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    p0 = parts[0].strip().rstrip(',')
                    p1 = parts[1].strip()
                    if len(p0.split()) > 5 and len(p1.split()) > 3:
                        p1 = p1[0].upper() + p1[1:]
                        s = p0 + '. ' + p1

            # Split very long sentence at "yang mana"
            if len(s.split()) > 22:
                parts = re.split(r'\byang mana\b', s, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    p0 = parts[0].strip().rstrip(',')
                    p1 = parts[1].strip()
                    if len(p0.split()) > 5 and len(p1.split()) > 3:
                        p1 = p1[0].upper() + p1[1:]
                        s = p0 + '. ' + p1

            new_sentences.append(s)

        result_paragraphs.append(' '.join(new_sentences))

    return '\n'.join(result_paragraphs)


def _inject_short_sentences(text: str, lang: str, style_mode: str = "populer") -> str:
    injects = {
        "akademik": {
            "id": ["Menariknya.", "Perlu dicatat.", 
                   "Ini krusial.", "Patut diperhatikan."],
            "en": ["Notably.", "Worth noting.", 
                   "This matters.", "Interestingly."]
        },
        "profesional": {
            "id": ["Faktanya.", "Singkat kata.", 
                   "Konkretnya.", "Perlu diperhatikan."],
            "en": ["In fact.", "Simply put.", 
                   "Concretely.", "Worth noting."]
        },
        "kreatif": {
            "id": ["Diam.", "Sunyi.", "Tiba-tiba.", 
                   "Aneh.", "Betul juga."],
            "en": ["Silence.", "Strange.", 
                   "Suddenly.", "Indeed."]
        },
        "populer": {
            "id": [
                "Menarik.", "Sungguh.", "Wajar saja.",
                "Memang begitu.", "Cukup rumit.", "Tentu.",
                "Tidak mudah.", "Perlu diakui."
            ],
            "en": [
                "Interesting.", "Indeed.", "Fair enough.",
                "That said.", "Not easy.", "Worth noting."
            ]
        }
    }
    lang_key = "id" if lang in ("id", "mixed") else "en"
    short_list = injects.get(style_mode, injects["populer"])[lang_key]
    
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
                new_sentences.append(random.choice(short_list))
        result.append(' '.join(new_sentences))
    return '\n'.join(result)


def _replace_preserve_case(text: str, pattern: str, replacement: str) -> str:
    """Replace pattern tapi preserve kapitalisasi kata pertama."""
    def _replacer(match):
        orig = match.group(0)
        if orig and orig[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement
    return re.sub(pattern, _replacer, text, flags=re.IGNORECASE)

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
        if style_mode in ("akademik", "profesional"):
            # Mode formal: ganti hanya frasa AI yang paling generik,
            # JANGAN ganti kata baku yang membentuk kalimat benar
            replacements = [
                # Frasa AI opener yang bisa diganti tanpa melanggar EYD
                (r"(?i)\bsecara keseluruhan\s*,?\s*", "secara umum,"),
                (r"(?i)\bdapat disimpulkan bahwa\b", "dapat dikatakan bahwa"),
                (r"(?i)\bperlu dicatat bahwa\b", "perlu diperhatikan bahwa"),
                (r"(?i)\bdi sisi lain\s*,?\s*", "sebaliknya,"),
                (r"(?i)\bsangat penting\b", "krusial"),
                (r"(?i)\bhal ini menunjukkan bahwa\b", "ini mengindikasikan bahwa"),
                (r"(?i)\bhal ini membuktikan\b", "ini membuktikan"),
                (r"(?i)\btersebut\b", "ini"),
                # JANGAN ganti: adalah, merupakan, memiliki, berdasarkan,
                # terhadap, sehingga — semua itu kata baku yang benar
            ]
            conversational_injects = [
                "Perlu dicatat, ", "Menariknya, ", 
                "Di sisi lain, ", "Lebih jauh, "
            ]
        elif style_mode == "kreatif":
            replacements = [
                (r"(?i)\bsecara keseluruhan\s*,?\s*", "pada akhirnya,"),
                (r"(?i)\bdapat disimpulkan bahwa\b", "ternyata"),
                (r"(?i)\bhal ini\b", "ini"),
                (r"(?i)\btersebut\b", "itu"),
                (r"(?i)\bsangat\b", "amat"),
                (r"(?i)\bmenimbulkan\b", "melahirkan"),
                (r"(?i)\bsehingga\b", "hingga"),
            ]
            conversational_injects = [
                "Dan ", "Tapi ", "Tiba-tiba, ", "Sayangnya, "
            ]
        else:  # populer — conversational tapi TETAP EYD
            replacements = [
                # Ganti frasa AI dengan frasa natural EYD (bukan slang)
                (r"(?i)\bsecara keseluruhan\s*,?\s*", "singkat kata,"),
                (r"(?i)\bselain itu\s*,?\s*", "selain itu juga,"),
                (r"(?i)\bkesimpulannya\s*,?\s*", "intinya,"),
                (r"(?i)\boleh karena itu\s*,?\s*", "karena itu,"),
                (r"(?i)\bdapat disimpulkan bahwa\b", "bisa dikatakan bahwa"),
                (r"(?i)\bsangat penting\b", "sangat krusial"),
                (r"(?i)\bhal ini\b", "ini"),
                (r"(?i)\btersebut\b", "itu"),
                (r"(?i)\bmenimbulkan\b", "menimbulkan"),
                (r"(?i)\bnamun demikian\s*,?\s*", "namun,"),
                (r"(?i)\bdengan demikian\s*,?\s*", "dengan begitu,"),
                # HAPUS: adalah→itu, merupakan→itu (melanggar EYD)
                # HAPUS: terhadap→pada (tidak selalu benar)
                # HAPUS: berbagai→macam-macam (tidak selalu natural)
                (r"(?i)\bberbagai macam\b", "berbagai"),  # hanya ini yang aman
            ]
            conversational_injects = [
                "Dan ", "Tapi ", "Nah, ", "Memang, ", "Tentu saja, "
            ]
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
        text = _replace_preserve_case(text, pattern, replacement)
    
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
                    sent_lower = sentences[i].lower()
                    # Skip jika kalimat sudah dimulai connector/transisi
                    already_has_connector = any(
                        sent_lower.startswith(c.lower()) 
                        for c in ["dan ", "tapi ", "nah,", "memang,", 
                                  "tentu", "perlu", "menarik", "di sisi",
                                  "lebih jauh", "karena itu", "dengan begitu",
                                  "selain", "namun", "bahkan", "justru"]
                    )
                    if not already_has_connector:
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
        f"Tulis ulang draf berikut agar terdengar natural, "
        f"ditulis oleh manusia sungguhan. "
        f"Ikuti register dan gaya yang telah ditentukan. "
        f"Output HARUS persis {paragraph_count} paragraf.\n\n"
        f"{clean_draft}"
    )
    
    async with agent.run_stream(
        user_msg,
        model_settings={"temperature": 1.4}
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
    text = _programmatic_sentence_humanize(text, style.language, style_mode)
    
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
        f"Tulis ulang draf berikut agar terdengar natural, "
        f"ditulis oleh manusia sungguhan. "
        f"Ikuti register dan gaya yang telah ditentukan. "
        f"Output HARUS persis {paragraph_count} paragraf.\n\n"
        f"{clean_draft}"
    )
    
    result = await agent.run(user_msg, model_settings={"temperature": 1.4})
    text = result.output.final_text
    
    text = _apply_post_processing(text, style.language, style_mode)
    text = _inject_short_sentences(text, style.language, style_mode)
    text = _programmatic_sentence_humanize(text, style.language, style_mode)
    text = _apply_post_processing(text, style.language, style_mode)
    
    result.output.final_text = text
    return result.output

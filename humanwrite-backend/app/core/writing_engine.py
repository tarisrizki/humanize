"""Writing Engine — Rewrites draft text to match a user's StyleProfile.

Uses PydanticAI with Groq to perform intelligent rewriting that evades AI detection.
"""

import asyncio
import json
import os
import re
import random
import statistics
from typing import AsyncGenerator

from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from langdetect import detect

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


def _build_system_prompt(style: StyleProfile, paragraph_count: int, input_lang: str) -> str:
    style_mode = getattr(style, 'style_mode', 'populer')

    if input_lang == "id":
        lang_instruction = "Tulis dalam Bahasa Indonesia yang baik dan benar sesuai EYD."
        active_few_shots = style.few_shot_examples_id
    else:
        lang_instruction = "Write in natural English."
        active_few_shots = style.few_shot_examples_en

    few_shot_text = ""
    if active_few_shots:
        examples_str = "\n\n".join([
            f"Contoh {i+1}:\n{ex}"
            for i, ex in enumerate(active_few_shots)
        ])
        few_shot_text = f"""
## PANDUAN GAYA PENULISAN (WAJIB DIIKUTI)
Berikut contoh nyata tulisan yang menjadi acuan.
Tiru irama kalimat, pilihan kata, dan alur narasinya.

{examples_str}
"""

    if style_mode == "akademik":
        register_rules = """
## REGISTER: AKADEMIK — Formal, ilmiah, baku, EYD.

BOLEH:
- Kata formal: merupakan, adalah, berdasarkan, sehingga, 
  mengindikasikan, menunjukkan, dilakukan, diperoleh.
- Kalimat pasif (lazim dalam tulisan ilmiah).
- Hedging: "tampaknya", "cenderung menunjukkan", 
  "mengindikasikan bahwa", "dapat diasumsikan".
- Catatan analitis pendek: "Ini signifikan.", 
  "Perlu dicermati.", "Menarik untuk dikaji."

HINDARI MUTLAK:
- Bahasa informal: "masih ingat kan?", "bukan?", 
  "sungguh kompleks", "semoga saja", "ya kan".
- Pertanyaan retoris informal.
- Kalimat baru yang tidak ada di draf asli.

BURSTINESS (WAJIB):
- Setiap paragraf HARUS mengandung minimal:
  * 1 kalimat analitis pendek (5-8 kata): 
    "Ini signifikan.", "Angka ini mengkhawatirkan.", 
    "Dampaknya luas."
  * 1 kalimat panjang dan detail (20-30 kata).
- Jangan semua kalimat seragam panjangnya.
"""
    elif style_mode == "profesional":
        register_rules = """
## REGISTER: PROFESIONAL — Formal, ringkas, action-oriented.

BOLEH:
- Bahasa fungsional baku, to-the-point.
- Kalimat aktif yang tegas.
- Catatan pendek: "Faktanya.", "Ini penting.", "Perlu diperhatikan."

HINDARI:
- Jargon berlebihan, frasa AI generik.
- Kalimat baru yang tidak ada di draf asli.

BURSTINESS (WAJIB):
- Mix kalimat tegas pendek (3-6 kata) dengan 
  kalimat informatif panjang (18-25 kata).
- Jangan semua kalimat seragam.
"""
    elif style_mode == "kreatif":
        register_rules = """
## REGISTER: KREATIF — Ekspresif, naratif, emosional.

BOLEH:
- Diksi vivid, metafora, detail sensoris.
- Kalimat sangat pendek (2-4 kata) dan sangat panjang (30+ kata).
- Mulai kalimat dengan "Dan", "Tapi", "Meski", "Karena".

HINDARI:
- Frasa klise AI, penjelasan terlalu eksplisit.
- Kalimat baru yang tidak ada di draf asli.

BURSTINESS (WAJIB):
- Variasi panjang kalimat EKSTREM: 2-3 kata hingga 35 kata.
- Show, don't tell.
"""
    else:  # populer
        register_rules = """
## REGISTER: POPULER — Mudah dipahami, mengalir, EYD.

BOLEH:
- Bahasa conversational namun tetap baku dan EYD.
- Kalimat pendek tegas: "Ini penting.", "Tidak mudah.", 
  "Wajar saja.", "Cukup rumit."

HINDARI:
- Bahasa gaul, slang, tidak EYD.
- "hal ini", "tersebut", "dapat disimpulkan", 
  "secara keseluruhan", "sangat penting".
- Kalimat baru yang tidak ada di draf asli.

BURSTINESS (WAJIB):
- Mix kalimat pendek (3-5 kata) dan kalimat panjang 
  yang mengalir (18-25 kata) dalam setiap paragraf.
"""

    anti_plagiarism_block = """
## ANTI-PLAGIARISME (WAJIB — SAMA PENTINGNYA DENGAN ANTI-AI)
Teks ini juga akan diuji Turnitin. Pastikan:

1. TIDAK BOLEH ada 3 kata berurutan yang SAMA PERSIS
   dengan draf asli. Ini aturan mutlak tanpa pengecualian.

2. WAJIB UBAH STRUKTUR KALIMAT, bukan hanya ganti kata:
   - Kalimat aktif → pasif, atau pasif → aktif
   - Gabungkan 2 kalimat pendek menjadi 1 kalimat kompleks
   - Pecah 1 kalimat panjang (>25 kata) menjadi 2 kalimat
   - Pindah posisi klausa: yang di awal → pindah ke akhir kalimat
   - Ubah urutan informasi dalam paragraf

3. HINDARI frasa akademik generik yang ada di jutaan dokumen:
   ID: "penelitian ini bertujuan", "berdasarkan hasil",
       "dapat disimpulkan bahwa", "perlu dilakukan",
       "dalam rangka", "guna mencapai", "sebagai berikut"
   EN: "this study aims to", "based on the results",
       "it can be concluded", "further research is needed",
       "in order to", "as follows", "it is important to note"
   Ganti dengan konstruksi kalimat yang lebih spesifik.

4. VARIASIKAN panjang kalimat secara ekstrem — ini membantu
   bypass AI detector sekaligus membuat teks unik secara
   struktural sehingga tidak cocok dengan sumber manapun.
"""

    return f"""Kamu penulis manusia berpengalaman, bukan AI.
Tugas: tulis ulang draf agar tidak terdeteksi detektor AI
seperti GPTZero, sambil mempertahankan semua fakta asli.

{lang_instruction}

{few_shot_text}

{anti_plagiarism_block}

{register_rules}

## ATURAN WAJIB (TIDAK BOLEH DILANGGAR)

1. BAHASA: {lang_instruction}

2. KONTEN ASLI WAJIB DIPERTAHANKAN: setiap kalimat output
   harus merupakan parafrasa dari kalimat di draf asli.
   DILARANG menambah opini, pertanyaan, atau informasi
   yang tidak ada di draf.

3. STRUKTUR PARAGRAF WAJIB DIPERTAHANKAN:
   - Draf asli: {paragraph_count} paragraf.
   - Output WAJIB: persis {paragraph_count} paragraf.
   - Pisahkan dengan baris kosong antar paragraf.
   - JANGAN gabungkan atau pisahkan paragraf.
   - INI ATURAN PALING PENTING. PELANGGARAN TIDAK DIIZINKAN.

4. BURSTINESS: ikuti instruksi register di atas.
   Variasi panjang kalimat adalah kunci output berkualitas.

5. TIDAK ADA FRASA AI: hindari "hal ini menunjukkan bahwa",
   "dapat disimpulkan", "secara keseluruhan", "sangat penting",
   "selain itu", "oleh karena itu", "dengan demikian",
   "tersebut", "memiliki", "berbagai", "merupakan".
   Ganti dengan parafrasa yang lebih natural sesuai register.

6. Catat perubahan signifikan di changes_made.
"""



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

            # Split kalimat >25 kata di titik natural
            if len(s.split()) > 25:
                # Coba split di "yang" jika ada subjek sebelumnya
                parts = re.split(r',?\s+yang\s+', s, maxsplit=1)
                if len(parts) == 2 and len(parts[0].split()) >= 6:
                    p1 = parts[0].strip().rstrip(',') + '.'
                    p2 = parts[1].strip()
                    if p2:
                        p2 = p2[0].upper() + p2[1:]
                        # Tambah konteks agar p2 bisa berdiri sendiri
                        s = p1 + ' ' + p2
            
            # Split di "dan" untuk kalimat compound >28 kata
            if len(s.split()) > 28:
                # Hanya split jika "dan" ada di tengah (bukan awal/akhir)
                mid = len(s) // 2
                search_zone = s[mid-30:mid+30]
                dan_match = re.search(r'\bdan\b', search_zone)
                if dan_match:
                    abs_pos = mid - 30 + dan_match.start()
                    part1 = s[:abs_pos].strip().rstrip(',')
                    part2 = s[abs_pos+3:].strip()
                    if len(part1.split()) >= 8 and len(part2.split()) >= 5:
                        part2 = part2[0].upper() + part2[1:]
                        s = part1 + '. ' + part2

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
            if len(sent.split()) > 20 and random.random() < 0.10:
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
                (r"(?i)\btersebut\b", "itu"),
                (r"(?i)\bsangat\b", "amat"),
                (r"(?i)\bmenimbulkan\b", "melahirkan"),
                (r"(?i)\bsehingga\b", "hingga"),
            ]
            conversational_injects = [
                "Dan ", "Tapi ", "Meski begitu, ", 
                "Anehnya, ", "Yang jelas, ", "Sayangnya, "
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
                        for c in [
                            "dan ", "tapi ", "nah,", "memang,", "tentu",
                            "perlu", "menarik", "di sisi", "lebih jauh",
                            "karena itu", "dengan begitu", "selain", "namun",
                            "bahkan", "justru", "oleh karena", "dengan demikian",
                            "sementara", "adapun", "terkait", "di samping",
                            "pada akhirnya", "singkat kata", "intinya",
                            "buktinya", "ternyata", "maka,", "anehnya",
                            "sayangnya", "secara umum", "dalam konteks",
                            "berdasarkan", "di samping itu", "meski begitu",
                            "yang jelas", "kondisi ini", "temuan ini",
                            "data ini", "dapat dikatakan", "perlu diperhatikan",
                        ]
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

def check_trigram_overlap(original: str, rewritten: str) -> float:
    """
    Hitung persentase trigram (3 kata berurutan) yang sama
    antara original dan rewritten.
    Target aman Turnitin: < 15%
    Return: float 0.0 - 1.0
    """
    def get_trigrams(text: str) -> set:
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        if len(words) < 3:
            return set()
        return set(zip(words, words[1:], words[2:]))

    orig_trigrams = get_trigrams(original)
    new_trigrams  = get_trigrams(rewritten)

    if not orig_trigrams:
        return 0.0

    overlap = len(orig_trigrams & new_trigrams)
    return round(overlap / len(orig_trigrams), 3)


def _needs_rewrite(overlap: float, threshold: float = 0.30) -> bool:
    """
    Return True jika trigram overlap terlalu tinggi
    dan teks perlu Pass 2 rewrite.
    Threshold default 30% — agresif tapi tidak berlebihan.
    """
    return overlap > threshold

def _validate_paragraph_count(
    text: str,
    expected: int,
    original_draft: str
) -> str:
    """Pastikan jumlah paragraf output sesuai draf asli."""
    actual = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(actual) == expected:
        return text

    # Fallback: jika output jadi 1 paragraf tapi harusnya lebih
    if len(actual) == 1 and expected > 1:
        orig_paras = [p.strip() for p in original_draft.split('\n\n') if p.strip()]
        all_sents = re.split(r'(?<=[.!?])\s+', text.strip())

        orig_counts = [
            max(1, len(re.split(r'(?<=[.!?])\s+', op.strip())))
            for op in orig_paras
        ]
        total_orig = sum(orig_counts)
        total_new = len(all_sents)

        result = []
        idx = 0
        for i, count in enumerate(orig_counts):
            if i == len(orig_counts) - 1:
                result.append(' '.join(all_sents[idx:]))
            else:
                take = max(1, round(count / total_orig * total_new))
                result.append(' '.join(all_sents[idx:idx + take]))
                idx += take

        return '\n\n'.join(p for p in result if p.strip())

    return text

def _generate_changes_made(
    original: str,
    rewritten: str,
    trigram_overlap: float = None,
) -> list[str]:
    """
    Generate changes_made secara programatik dari diff
    antara original dan rewritten text.
    Tidak butuh LLM — lebih reliable dan bebas constraint.
    """
    changes = []
    
    orig_sents = re.split(r'(?<=[.!?])\s+', original.strip())
    new_sents  = re.split(r'(?<=[.!?])\s+', rewritten.strip())
    
    # Cek perubahan jumlah kalimat
    delta = len(new_sents) - len(orig_sents)
    if abs(delta) >= 2:
        if delta > 0:
            changes.append(
                f"Beberapa kalimat panjang dipecah menjadi kalimat lebih pendek "
                f"untuk meningkatkan variasi ritme."
            )
        else:
            changes.append(
                f"Beberapa kalimat pendek digabungkan untuk alur yang lebih baik."
            )
    
    # Cek kata AI yang dihilangkan
    ai_words = {
        "merupakan", "memiliki", "berbagai", "sehingga", "serta",
        "tersebut", "selain itu", "oleh karena itu", "dengan demikian",
        "hal ini", "dapat disimpulkan", "secara keseluruhan",
        "sangat penting", "perlu dicatat", "di sisi lain",
    }
    orig_lower = original.lower()
    new_lower  = rewritten.lower()
    removed_ai = [w for w in ai_words if w in orig_lower and w not in new_lower]
    if removed_ai:
        changes.append(
            f"Frasa AI generik dihapus/diganti: "
            f"{', '.join(f'\"{w}\"' for w in removed_ai[:4])}."
        )
    
    # Cek variasi panjang kalimat
    if orig_sents and new_sents:
        orig_lens = [len(s.split()) for s in orig_sents]
        new_lens  = [len(s.split()) for s in new_sents]
        orig_std = statistics.stdev(orig_lens) if len(orig_lens) > 1 else 0
        new_std  = statistics.stdev(new_lens)  if len(new_lens)  > 1 else 0
        if new_std > orig_std + 2:
            changes.append(
                "Variasi panjang kalimat ditingkatkan untuk ritme yang lebih natural "
                "(burstiness)."
            )
    
    # Fallback
    if not changes:
        changes.append(
            "Parafrasa kalimat untuk gaya penulisan yang lebih natural "
            "dan tidak terdeteksi AI."
        )
    
    if trigram_overlap is not None:
        safety = (
            "aman" if trigram_overlap < 0.15
            else "perlu perhatian" if trigram_overlap < 0.30
            else "risiko tinggi"
        )
        changes.append(
            f"Kesamaan struktural dengan draf asli: "
            f"{trigram_overlap:.0%} ({safety} untuk Turnitin)."
        )

    return changes


async def apply_style_stream(
    draft: str, style: StyleProfile
) -> AsyncGenerator[str, None]:
    clean_draft = _clean_input_draft(draft)
    paragraph_count = _count_paragraphs(clean_draft)
    
    try:
        input_lang = "id" if detect(clean_draft[:2000]) in ("id", "ms") else "en"
    except Exception:
        input_lang = "id"
        
    system_prompt = _build_system_prompt(style, paragraph_count, input_lang)
    style_mode = getattr(style, 'style_mode', 'populer')

    # Plain text agent — TANPA output_type
    # Ini membebaskan Groq menulis prose alami tanpa constraint JSON
    agent = Agent(
        model=FallbackModel(
            "groq:llama-3.3-70b-versatile",
            "groq:llama-3.1-8b-instant",
        ),
        system_prompt=system_prompt,
        # TIDAK ADA output_type=ProcessedText
    )

    user_msg = (
        f"Tulis ulang draf berikut agar terdengar natural, "
        f"ditulis oleh manusia sungguhan. "
        f"Ikuti register dan gaya yang telah ditentukan. "
        f"Kembalikan HANYA teks hasil rewrite — "
        f"jangan tambahkan penjelasan, label, atau komentar apapun. "
        f"Output WAJIB persis {paragraph_count} paragraf "
        f"dipisahkan baris kosong.\n\n"
        f"{clean_draft}"
    )

    try:
        result = await asyncio.wait_for(
            agent.run(
                user_msg,
                model_settings={"temperature": 1.5},
            ),
            timeout=60.0
        )
        full_text = str(result.output).strip() if result.output else ""
    except Exception as e:
        full_text = ""

    if not full_text:
        return

    # Post-processing pipeline
    text = _apply_post_processing(full_text, style.language, style_mode)
    text = _inject_short_sentences(text, style.language, style_mode)
    text = _programmatic_sentence_humanize(text, style.language, style_mode)
    text = _apply_post_processing(text, style.language, style_mode)
    text = _validate_paragraph_count(text, paragraph_count, clean_draft)

    # ── Trigram check — Pass 2 jika overlap terlalu tinggi ───
    trigram_overlap = check_trigram_overlap(clean_draft, text)

    if _needs_rewrite(trigram_overlap, threshold=0.30):
        # Overlap terlalu tinggi → minta Groq rewrite lagi
        # dengan instruksi structural yang lebih agresif
        pass2_msg = (
            f"Teks berikut masih memiliki {trigram_overlap:.0%} kesamaan "
            f"struktural dengan draf asli. "
            f"Tulis ulang SELURUH teks dengan mengubah struktur setiap "
            f"kalimat secara fundamental — ubah urutan subjek-predikat-objek, "
            f"gabungkan atau pecah kalimat, ubah aktif ke pasif atau sebaliknya. "
            f"TIDAK BOLEH ada 3 kata berurutan yang sama dengan versi ini.\n\n"
            f"{text}"
        )
        try:
            result2 = await asyncio.wait_for(
                agent.run(
                    pass2_msg,
                    model_settings={"temperature": 1.5},
                ),
                timeout=60.0
            )
            text2 = str(result2.output).strip() if result2.output else ""
            if text2:
                text2 = _apply_post_processing(text2, style.language, style_mode)
                text2 = _programmatic_sentence_humanize(text2, style.language, style_mode)
                text2 = _validate_paragraph_count(text2, paragraph_count, clean_draft)
                # Hanya pakai Pass 2 jika overlap-nya lebih kecil
                new_overlap = check_trigram_overlap(clean_draft, text2)
                if new_overlap < trigram_overlap:
                    text = text2
                    trigram_overlap = new_overlap
        except Exception:
            pass  # Tetap pakai Pass 1 jika Pass 2 gagal

    # Simulated streaming
    chunk_size = 10
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        yield f"event: text\ndata: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)

    # Changes programmatic — tidak dari LLM
    changes = _generate_changes_made(clean_draft, text, trigram_overlap)
    yield f"event: metrics\ndata: {json.dumps({'changes_made': changes})}\n\n"


async def apply_style(
    draft: str, style: StyleProfile
) -> ProcessedText:
    clean_draft = _clean_input_draft(draft)
    paragraph_count = _count_paragraphs(clean_draft)
    
    try:
        input_lang = "id" if detect(clean_draft[:2000]) in ("id", "ms") else "en"
    except Exception:
        input_lang = "id"
        
    system_prompt = _build_system_prompt(style, paragraph_count, input_lang)
    style_mode = getattr(style, 'style_mode', 'populer')

    agent = Agent(
        model=FallbackModel(
            "groq:llama-3.3-70b-versatile",
            "groq:llama-3.1-8b-instant",
        ),
        system_prompt=system_prompt,
        # TIDAK ADA output_type=ProcessedText
    )

    user_msg = (
        f"Tulis ulang draf berikut agar terdengar natural, "
        f"ditulis oleh manusia sungguhan. "
        f"Ikuti register dan gaya yang telah ditentukan. "
        f"Kembalikan HANYA teks hasil rewrite — "
        f"jangan tambahkan penjelasan, label, atau komentar apapun. "
        f"Output WAJIB persis {paragraph_count} paragraf "
        f"dipisahkan baris kosong.\n\n"
        f"{clean_draft}"
    )

    try:
        result = await asyncio.wait_for(
            agent.run(
                user_msg,
                model_settings={"temperature": 1.5},
            ),
            timeout=60.0
        )
        full_text = str(result.output).strip() if result.output else ""
    except Exception:
        full_text = draft  # fallback ke draf asli

    text = _apply_post_processing(full_text, style.language, style_mode)
    text = _inject_short_sentences(text, style.language, style_mode)
    text = _programmatic_sentence_humanize(text, style.language, style_mode)
    text = _apply_post_processing(text, style.language, style_mode)
    text = _validate_paragraph_count(text, paragraph_count, clean_draft)

    # ── Trigram check — Pass 2 jika overlap terlalu tinggi ───
    trigram_overlap = check_trigram_overlap(clean_draft, text)

    if _needs_rewrite(trigram_overlap, threshold=0.30):
        pass2_msg = (
            f"Teks berikut masih memiliki {trigram_overlap:.0%} kesamaan "
            f"struktural dengan draf asli. "
            f"Tulis ulang SELURUH teks dengan mengubah struktur setiap "
            f"kalimat secara fundamental — ubah urutan subjek-predikat-objek, "
            f"gabungkan atau pecah kalimat, ubah aktif ke pasif atau sebaliknya. "
            f"TIDAK BOLEH ada 3 kata berurutan yang sama dengan versi ini.\n\n"
            f"{text}"
        )
        try:
            result2 = await asyncio.wait_for(
                agent.run(
                    pass2_msg,
                    model_settings={"temperature": 1.5},
                ),
                timeout=60.0
            )
            text2 = str(result2.output).strip() if result2.output else ""
            if text2:
                text2 = _apply_post_processing(text2, style.language, style_mode)
                text2 = _programmatic_sentence_humanize(text2, style.language, style_mode)
                text2 = _validate_paragraph_count(text2, paragraph_count, clean_draft)
                new_overlap = _check_trigram_overlap(clean_draft, text2)
                if new_overlap < trigram_overlap:
                    text = text2
                    trigram_overlap = new_overlap
        except Exception:
            pass

    changes = _generate_changes_made(clean_draft, text, trigram_overlap)
    return ProcessedText(final_text=text, changes_made=changes)

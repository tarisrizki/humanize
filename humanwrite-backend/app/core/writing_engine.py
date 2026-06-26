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


def _build_system_prompt(style: StyleProfile, paragraph_count: int) -> str:
    style_mode = getattr(style, 'style_mode', 'populer')

    if style.language == "id":
        lang_instruction = (
            "Tulis HANYA dalam Bahasa Indonesia yang baik dan benar. "
            "EYD wajib dipatuhi. Jangan campur dengan bahasa lain."
        )
    elif style.language == "mixed":
        lang_instruction = (
            "Pertahankan pola campur kode asli dari draf — "
            "jika draf Indonesia, output Indonesia. "
            "Jika draf campuran, pertahankan rasio yang sama."
        )
    else:
        lang_instruction = "Write in natural, idiomatic English only."

    few_shot_text = ""
    if style.language in ("id", "mixed"):
        examples = getattr(style, 'few_shot_examples_id', []) or \
                   getattr(style, 'few_shot_examples', [])
    else:
        examples = getattr(style, 'few_shot_examples_en', []) or \
                   getattr(style, 'few_shot_examples', [])

    if examples:
        examples_str = "\n\n---\n\n".join(examples[:4])
        few_shot_text = f"""
## PANDUAN GAYA — WAJIB DITIRU
Berikut contoh tulisan manusia nyata. Tiru irama, 
panjang kalimat, pilihan kata, dan alur paragrafnya.
Ini adalah standar yang harus dicapai:

{examples_str}

---
"""

    if style_mode == "akademik":
        register_rules = f"""
## REGISTER: AKADEMIK

STRUKTUR PARAGRAF WAJIB (minimum {paragraph_count} paragraf):
Setiap paragraf HARUS mengandung kelima elemen ini:
1. Kalimat pembuka yang tidak klise (10-20 kata)
2. Kalimat analitis panjang dengan nuansa (22-32 kata) 
3. Kalimat medium yang memperkuat (12-18 kata)
4. Kalimat catatan pendek yang mengejutkan (4-8 kata) 
   — contoh: "Angka ini cukup mengkhawatirkan." 
              "Temuan ini tidak bisa diabaikan."
              "Ironisnya, justru sebaliknya."
5. Kalimat penutup yang membuka pertanyaan baru 
   atau memberi perspektif segar (12-20 kata)

WAJIB DIGUNAKAN — HEDGING AKADEMIK NATURAL:
Sisipkan salah satu per paragraf (jangan berulang):
"tampaknya", "agaknya", "cenderung menunjukkan bahwa",
"boleh jadi", "sejauh kajian ini menjangkau",
"data mengisyaratkan", "pola ini mengarah pada",
"tidak menutup kemungkinan bahwa"

WAJIB DIGUNAKAN — OBSERVASI ANALITIS PERSONAL:
Satu kalimat per paragraf yang menunjukkan penulis 
sedang berpikir, bukan hanya melaporkan:
Contoh: "Yang menarik dari pola ini adalah..."
        "Paradoks ini layak mendapat perhatian lebih."
        "Situasi ini mencerminkan tantangan yang lebih dalam."

AKTIF vs PASIF: Variasikan — jangan semua pasif atau 
semua aktif. Ganti setiap 2-3 kalimat.

HINDARI: paragraf yang semua kalimatnya sama panjang,
pembuka paragraf yang berurutan dengan kata yang sama,
frasa yang terlalu sempurna dan terstruktur.
"""

    elif style_mode == "profesional":
        register_rules = f"""
## REGISTER: PROFESIONAL

STRUKTUR PARAGRAF WAJIB (minimum {paragraph_count} paragraf):
Setiap paragraf HARUS mengandung elemen ini:
1. Kalimat pembuka yang langsung ke substansi (8-15 kata)
2. Kalimat penjelas dengan data/fakta konkret (15-25 kata)
3. Kalimat tegas dan pendek yang menunjukkan kepastian 
   (4-8 kata): "Ini tidak bisa ditunda." / "Risikonya nyata."
4. Kalimat elaborasi atau konteks (12-20 kata)
5. Kalimat penutup yang actionable atau forward-looking

WAJIB: Setiap paragraf harus punya minimal 1 angka, 
nama spesifik, atau referensi konkret dari draf asli.

WAJIB: Gunakan kalimat aktif yang tegas dan kuat.
Contoh yang SALAH: "Hal tersebut perlu dipertimbangkan"
Contoh yang BENAR: "Tim harus mempertimbangkan ini segera."

VARIASI STRUKTUR: Jangan semua kalimat deklaratif. 
Sesekali gunakan kalimat implikatif atau kondisional:
"Jika ini dibiarkan...", "Tanpa intervensi segera..."

HINDARI: kalimat pasif berlebihan, frasa pengantar 
yang panjang sebelum inti informasi, eufemisme yang 
tidak perlu.
"""

    elif style_mode == "kreatif":
        register_rules = f"""
## REGISTER: KREATIF

STRUKTUR PARAGRAF (minimum {paragraph_count} paragraf):
TIDAK ADA template. Tapi setiap paragraf wajib punya:
- Minimal 1 kalimat sangat pendek (2-5 kata)
- Minimal 1 kalimat sangat panjang (25-40 kata)
- Minimal 1 elemen sensoris yang SPESIFIK dan tidak 
  generik (bukan "harum bunga" tapi aroma apa tepatnya,
  bukan "suara bising" tapi suara apa)

WAJIB — IMPERFEKSI NATURAL:
- Boleh ada kalimat yang pivot di tengah atau 
  berubah arah: "Ia bermaksud pergi — tapi kemudian..."
- Boleh ada kalimat fragment untuk efek dramatis: 
  "Tidak ada yang menjawab. Sepi."
- Pikiran yang tidak selesai sempurna: "Entah kenapa,"
- Simile yang tidak terduga dan sedikit aneh

WAJIB — SUDUT PANDANG:
Selalu ada "suara" yang jelas — narator yang punya 
perspektif, bukan reporter yang netral.

DILARANG: deskripsi sensoris generik, narasi yang 
terlalu smooth dan terkontrol, setiap paragraf 
berstruktur sama.
"""

    else:  # populer
        register_rules = f"""
## REGISTER: POPULER

STRUKTUR PARAGRAF WAJIB (minimum {paragraph_count} paragraf):
Setiap paragraf HARUS:
1. Dimulai dengan cara yang berbeda dari paragraf 
   sebelumnya (jangan semua dengan kata penghubung)
2. Punya minimal 1 kalimat yang terasa "berbicara 
   langsung" ke pembaca
3. Punya variasi panjang ekstrem:
   - 1 kalimat pendek (3-6 kata): "Ini yang berbeda."
   - 1 kalimat panjang mengalir (18-28 kata)
   - Sisanya medium (8-15 kata)
4. Minimum 4 kalimat per paragraf

WAJIB — ELEMEN CONVERSATIONAL:
Sisipkan salah satu per 2 paragraf:
"Jujur saja,", "Kalau dipikir-pikir,", 
"Yang bikin menarik,", "Memang tidak mudah,",
"Tapi tunggu dulu —", "Faktanya justru sebaliknya."

WAJIB — OPINI RINGAN:
Satu kalimat per paragraf yang menunjukkan penulis 
punya pendapat, bukan hanya melaporkan fakta.
Contoh: "Dan menurut saya, inilah inti masalahnya."
        "Pilihan ini lebih masuk akal daripada yang 
         terlihat di permukaan."

HINDARI: transisi yang terlalu formal, kalimat yang 
semua sama panjang, tidak ada "suara" penulis.
"""

    return f"""
## IDENTITAS & TUJUAN UTAMA
Kamu penulis Indonesia berpengalaman. Tugas satu-satunya:
tulis ulang draf agar TIDAK terdeteksi GPTZero dan Turnitin.

## ATURAN ABSOLUT — BACA INI DULU SEBELUM APAPUN

ATURAN 1 — MINIMUM 4 KALIMAT PER PARAGRAF:
Setiap paragraf WAJIB minimal 4 kalimat. Jika paragraf
asli hanya 2-3 kalimat, perluas dengan kalimat analitis
atau observasi yang memperkaya — tapi jangan tambah fakta baru.

ATURAN 2 — VARIASI PANJANG EKSTREM DALAM TIAP PARAGRAF:
Setiap paragraf WAJIB punya kombinasi:
  • 1 kalimat sangat pendek (3-6 kata)
  • 1 kalimat sangat panjang (25-35 kata)  
  • Sisanya medium (10-18 kata)
Ini adalah aturan paling penting untuk lolos deteksi AI.

ATURAN 3 — KATA YANG DILARANG KERAS:
Jangan gunakan dalam bentuk apapun:
merupakan, memiliki, berbagai, sehingga, serta, tersebut,
selain itu, oleh karena itu, dengan demikian, hal ini,
dapat disimpulkan, secara keseluruhan, sangat penting,
perlu dicatat, dalam hal ini, adapun, tentunya, pastinya,
furthermore, moreover, additionally, utilize, leverage,
it is important, in conclusion, notably, straightforward.

ATURAN 4 — STRUKTUR PARAGRAF NON-TEMPLATE:
JANGAN gunakan: topic sentence → penjelasan → contoh → kesimpulan.
LAKUKAN: mulai dari perspektif/observasi, lalu bangun ke argumen,
atau mulai dari contoh spesifik, lalu generalisasi.

ATURAN 5 — OUTPUT HANYA TEKS:
Langsung mulai kalimat pertama. Tidak ada penjelasan,
tidak ada bullet points, tidak ada catatan perubahan.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{lang_instruction}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{few_shot_text}

{register_rules}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PENGINGAT AKHIR — SEBELUM KAMU MULAI MENULIS:
1. Sudahkah kamu pastikan setiap paragraf ≥4 kalimat?
2. Sudahkah setiap paragraf punya kalimat pendek (3-6 kata)?
3. Sudahkah kamu hindari SEMUA kata di ATURAN 3?
4. Apakah struktur paragrafmu NON-TEMPLATE?
Jika jawabannya tidak, tulis ulang dulu sebelum submit.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    return re.sub(pattern, _replacer, text, flags=re.IGNORECASE | re.MULTILINE)

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
        # ── Universal blacklist replacement (semua mode) ──
        universal_replacements = [
            # "merupakan" → ganti sesuai konteks
            (r'(?i)\bini merupakan\b', 'ini adalah'),
            (r'(?i)\byang merupakan\b', 'yang menjadi'),
            (r'(?i)\bmerupakan sebuah\b', 'adalah sebuah'),
            (r'(?i)\bmerupakan salah satu\b', 'menjadi salah satu'),
            (r'(?i)\bmerupakan\b', 'menjadi'),

            # "memiliki" → ganti dengan kata konkret
            (r'(?i)\btidak memiliki\b', 'tidak punya'),
            (r'(?i)\bmemiliki kemampuan\b', 'mampu'),
            (r'(?i)\bmemiliki peran\b', 'berperan'),
            (r'(?i)\bmemiliki dampak\b', 'berdampak'),
            (r'(?i)\bmemiliki\b', 'punya'),

            # "berbagai" → lebih spesifik
            (r'(?i)\bberbagai macam\b', 'beragam'),
            (r'(?i)\bberbagai\b', 'beragam'),

            # "sehingga" → variasi
            (r'(?i)\bsehingga\b', 'hingga'),

            # "serta" → lebih natural
            (r'(?i)\bserta\b', 'dan juga'),

            # "tersebut" → ganti dengan "ini" atau "itu"
            # Hati-hati: hanya jika tidak membingungkan
            (r'(?i)\bhal tersebut\b', 'hal ini'),
            (r'(?i)\bmasalah tersebut\b', 'masalah ini'),
            (r'(?i)\bprogram tersebut\b', 'program ini'),
            (r'(?i)\bkondisi tersebut\b', 'kondisi itu'),
            (r'(?i)\bsituasi tersebut\b', 'situasi itu'),

            # "hal ini" → lebih konkret
            (r'(?i)\bhal ini menunjukkan bahwa\b', 'ini menandakan bahwa'),
            (r'(?i)\bhal ini membuktikan\b', 'fakta ini membuktikan'),
            (r'(?i)\bhal ini\b', 'kondisi ini'),

            # "selain itu" → variasi
            (r'(?i)^selain itu,?\s*', 'Di samping itu, '),
            (r'(?i)\bselain itu,?\b', 'di samping itu,'),

            # "oleh karena itu" → lebih tegas
            (r'(?i)^oleh karena itu,?\s*', 'Karena itu, '),
            (r'(?i)\boleh karena itu\b', 'karena itu'),

            # "dengan demikian" → lebih natural
            (r'(?i)^dengan demikian,?\s*', 'Dengan begitu, '),
            (r'(?i)\bdengan demikian\b', 'dengan begitu'),

            # "dapat disimpulkan" → lebih langsung
            (r'(?i)\bdapat disimpulkan bahwa\b', 'intinya,'),
            (r'(?i)\bdapat disimpulkan\b', 'dapat dikatakan'),

            # "secara keseluruhan" → lebih segar
            (r'(?i)^secara keseluruhan,?\s*', 'Bila dilihat secara utuh, '),
            (r'(?i)\bsecara keseluruhan\b', 'secara menyeluruh'),

            # "sangat penting" → lebih kuat
            (r'(?i)\bsangat penting untuk\b', 'mendesak untuk'),
            (r'(?i)\bsangat penting\b', 'amat krusial'),

            # "perlu dicatat bahwa" → lebih natural
            (r'(?i)\bperlu dicatat bahwa\b', 'perlu diperhatikan,'),
            (r'(?i)\bperlu dicatat\b', 'patut dicermati'),

            # "dalam hal ini" → lebih spesifik
            (r'(?i)^dalam hal ini,?\s*', 'Dalam konteks ini, '),
            (r'(?i)\bdalam hal ini\b', 'dalam konteks ini'),

            # "adapun" → lebih natural
            (r'(?i)^adapun\b,?\s*', 'Sementara itu, '),
            (r'(?i)\badapun\b', 'sementara itu'),

            # "tentunya" dan "pastinya" → lebih subtle
            (r'(?i)\btentunya\b', 'tentu'),
            (r'(?i)\bpastinya\b', 'sudah pasti'),

            # "tidak hanya itu" → lebih mengalir
            (r'(?i)\btidak hanya itu\b', 'lebih dari itu'),

            # "di sisi lain" → variasi
            (r'(?i)^di sisi lain,?\s*', 'Sebaliknya, '),
            (r'(?i)\bdi sisi lain\b', 'sebaliknya'),
        ]

        # Apply universal replacements dengan preserve case
        for pattern, replacement in universal_replacements:
            text = _replace_preserve_case(text, pattern, replacement)

        # Bersihkan double comma yang mungkin muncul
        text = re.sub(r',\s*,', ',', text)
        # Bersihkan spasi double
        text = re.sub(r' {2,}', ' ', text)

        if style_mode == "akademik":
            replacements = [
                (r"(?i)\bsecara keseluruhan\s*,?\s*", "Menilik gambaran besarnya, "),
                (r"(?i)\bdapat disimpulkan bahwa\b", "pola ini mengarah pada"),
                (r"(?i)\bhal ini menunjukkan\b", "data mengisyaratkan"),
                (r"(?i)\bsangat penting\b", "krusial untuk dicermati"),
                (r"(?i)\btersebut\b", "ini"),
            ]
            conversational_injects = [
                "Menariknya, ", "Patut dicermati, ", 
                "Sebaliknya, ", "Lebih jauh, "
            ]
        elif style_mode == "profesional":
            replacements = [
                (r"(?i)\bsecara keseluruhan\s*,?\s*", "Secara menyeluruh, "),
                (r"(?i)\bhal ini menunjukkan\b", "data ini menegaskan"),
                (r"(?i)\bsangat penting\b", "mendesak"),
                (r"(?i)\btersebut\b", "ini"),
            ]
            conversational_injects = [
                "Menariknya, ", "Patut dicermati, ", 
                "Sebaliknya, ", "Lebih jauh, "
            ]
        elif style_mode == "kreatif":
            replacements = [
                (r"(?i)\bsecara keseluruhan\s*,?\s*", "Pada akhirnya, "),
                (r"(?i)\bhal ini\b", "semua ini"),
                (r"(?i)\bsangat\b", "benar-benar"),
                (r"(?i)\btersebut\b", "itu"),
                (r"(?i)\bmenimbulkan\b", "melahirkan"),
                (r"(?i)\bsehingga\b", "hingga"),
            ]
            conversational_injects = [
                "Dan ", "Tapi ", "Meski begitu, ", 
                "Anehnya, ", "Yang jelas, ", "Sayangnya, "
            ]
        else:  # populer
            replacements = [
                (r"(?i)\bsecara keseluruhan\s*,?\s*", "Kalau dipikir-pikir, "),
                (r"(?i)\bdapat disimpulkan bahwa\b", "intinya"),
                (r"(?i)\bhal ini\b", "situasi ini"),
                (r"(?i)\bsangat penting\b", "tidak bisa dianggap remeh"),
                (r"(?i)\bselain itu\s*,?\s*", "selain itu juga,"),
                (r"(?i)\boleh karena itu\s*,?\s*", "karena itu,"),
                (r"(?i)\btersebut\b", "itu"),
                (r"(?i)\bnamun demikian\s*,?\s*", "namun,"),
                (r"(?i)\bdengan demikian\s*,?\s*", "dengan begitu,"),
                (r"(?i)\bberbagai macam\b", "berbagai"),
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
                if random.random() < 0.08:  # reduced probability
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

def _strip_meta_commentary(text: str) -> str:
    """
    Hapus meta-commentary yang sering ditambahkan Groq:
    - Paragraf penjelasan di awal (sebelum teks asli)
    - Bullet point daftar perubahan di akhir
    - Kalimat transisi model tentang apa yang dilakukannya
    """
    lines = text.strip().split('\n')
    
    # Pola kalimat meta yang harus dihapus
    meta_patterns = [
        r'^teks di atas',
        r'^teks berikut',
        r'^berikut adalah teks',
        r'^berikut teks',
        r'^perubahan utama',
        r'^perubahan yang dilakukan',
        r'^dengan begitu',
        r'^demikianlah',
        r'^catatan:',
        r'^note:',
        r'^\*\s+mengganti',
        r'^\*\s+mengubah',
        r'^\-\s+mengganti',
        r'^\-\s+mengubah',
        r'^teks ulang sudah',
        r'^hasil teks ulang',
        r'^tapi berikut',
        r'^di bawah ini',
    ]
    
    cleaned_lines = []
    skip_bullet_section = False
    
    for line in lines:
        stripped = line.strip().lower()
        
        # Deteksi awal seksi bullet points
        if re.match(r'^perubahan', stripped) or \
           re.match(r'^catatan', stripped) or \
           re.match(r'^note:', stripped):
            skip_bullet_section = True
            continue
        
        # Skip bullet points
        if skip_bullet_section and (
            line.strip().startswith('*') or 
            line.strip().startswith('-') or
            line.strip().startswith('•')
        ):
            continue
        
        # Reset skip jika ada paragraf normal lagi
        if skip_bullet_section and len(line.strip()) > 50 and \
           not line.strip().startswith(('*', '-', '•')):
            skip_bullet_section = False
        
        # Cek apakah baris ini adalah meta-commentary
        is_meta = any(
            re.match(p, stripped) 
            for p in meta_patterns
        )
        
        if not is_meta:
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    return result if result else text


def _validate_output_quality(
    text: str,
    original: str,
    min_overlap: float = 0.15,
) -> bool:
    """
    Validasi apakah output Groq masih layak dipakai.
    Return False jika output terlalu menyimpang dari input.
    """
    if not text or len(text.split()) < 10:
        return False

    # Rasio panjang output vs input (harus 0.5x - 2.5x)
    orig_len = len(original.split())
    new_len  = len(text.split())
    ratio = new_len / max(orig_len, 1)
    if ratio < 0.5 or ratio > 2.5:
        return False

    # Minimal 15% kata penting (≥4 huruf) harus terjaga
    orig_words = set(re.findall(r'\b\w{4,}\b', original.lower()))
    new_words  = set(re.findall(r'\b\w{4,}\b', text.lower()))
    if not orig_words:
        return True
    overlap = len(orig_words & new_words) / len(orig_words)
    if overlap < min_overlap:
        return False

    # Tolak output yang mengandung meta-commentary
    text_lower = text.lower()
    meta_signals = [
        "perubahan utama",
        "perubahan yang dilakukan",
        "teks di atas masih",
        "berikut adalah teks ulang",
        "teks ulang sudah mengalami",
        "dengan begitu, teks ulang",
        "mengganti kalimat",
    ]
    if any(signal in text_lower for signal in meta_signals):
        return False

    # Tolak output yang mengandung terlalu banyak bullet points
    bullet_lines = sum(
        1 for l in text.split('\n')
        if l.strip().startswith(('* ', '- ', '• '))
    )
    if bullet_lines > 3:
        return False

    # Cek tidak ada kata yang sangat tidak lazim
    # (kata <3 huruf berturut-turut atau mengandung 
    #  konsonan >4 berturut-turut = kemungkinan hallucination)
    suspicious_words = [
        w for w in re.findall(r'\b\w+\b', text)
        if len(w) > 4 and re.search(r'[^aeiouAEIOU]{5,}', w)
        and w.lower() not in {'strength', 'through', 'straight',
                               'scream', 'spring', 'strong',
                               'kualitas', 'program', 'proses'}
    ]
    # Jika >3% kata mencurigakan → output buruk
    total_words = len(re.findall(r'\b\w+\b', text))
    if total_words > 0 and len(suspicious_words)/total_words > 0.03:
        return False

    return True

def _enforce_min_sentences(
    text: str,
    min_sentences: int = 4,
) -> str:
    """
    Cek setiap paragraf — jika kurang dari min_sentences,
    split kalimat panjang yang ada menjadi 2 kalimat.
    Ini enforcement terakhir sebelum output ke user.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    result = []

    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para.strip())
        sentences = [s for s in sentences if s.strip()]

        # Jika sudah cukup, tidak perlu diapa-apakan
        if len(sentences) >= min_sentences:
            result.append(' '.join(sentences))
            continue

        # Coba split kalimat terpanjang menjadi 2
        expanded = list(sentences)
        attempts = 0
        while len(expanded) < min_sentences and attempts < 3:
            # Cari kalimat terpanjang
            longest_idx = max(range(len(expanded)),
                            key=lambda i: len(expanded[i].split()))
            longest = expanded[longest_idx]
            words = longest.split()

            if len(words) < 12:
                break  # Tidak bisa displit lagi

            # Split di tengah pada titik natural
            # Cari "yang", "dan", "sehingga", "karena" di tengah
            mid = len(words) // 2
            split_words = ['yang', 'dan', 'karena', 'sehingga',
                          'namun', 'tetapi', 'meskipun', 'while',
                          'which', 'and', 'but', 'because']

            split_pos = None
            # Cari di zona tengah (40%-60% panjang kalimat)
            search_start = int(len(words) * 0.35)
            search_end   = int(len(words) * 0.65)
            for i in range(search_start, search_end):
                if words[i].lower() in split_words:
                    split_pos = i
                    break

            if split_pos:
                part1 = ' '.join(words[:split_pos]).rstrip(',') + '.'
                part2_words = words[split_pos:]
                # Capitalize first word of part 2
                if part2_words:
                    part2_words[0] = part2_words[0].capitalize()
                part2 = ' '.join(part2_words)
                if not part2.endswith(('.', '!', '?')):
                    part2 += '.'
                expanded[longest_idx:longest_idx+1] = [part1, part2]
            else:
                # Split di titik tengah paksa
                part1 = ' '.join(words[:mid]) + '.'
                part2 = ' '.join(words[mid:])
                if words[mid][0].islower():
                    part2 = part2[0].upper() + part2[1:]
                if not part2.endswith(('.', '!', '?')):
                    part2 += '.'
                expanded[longest_idx:longest_idx+1] = [part1, part2]

            attempts += 1

        result.append(' '.join(expanded))

    return '\n\n'.join(result)

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
        
    system_prompt = _build_system_prompt(style, paragraph_count)
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
        f"PENTING: setiap paragraf HARUS minimal 4 kalimat "
        f"dengan variasi panjang yang ekstrem. "
        f"Ikuti register dan gaya yang telah ditentukan. "
        f"Kembalikan HANYA teks hasil rewrite — "
        f"jangan tambahkan penjelasan, label, atau komentar. "
        f"Output WAJIB persis {paragraph_count} paragraf "
        f"dipisahkan baris kosong.\n\n"
        f"{clean_draft}"
    )

    try:
        result = await asyncio.wait_for(
            agent.run(
                user_msg,
                model_settings={"temperature": 1.2},
            ),
            timeout=60.0
        )
        full_text = str(result.output).strip() if result.output else ""
    except Exception as e:
        full_text = ""

    # Bersihkan meta-commentary sebelum validasi
    full_text = _strip_meta_commentary(full_text)

    # Validasi output Groq sebelum post-processing
    if not _validate_output_quality(full_text, clean_draft):
        # Output tidak layak — retry dengan temperature lebih rendah
        try:
            result_retry = await asyncio.wait_for(
                agent.run(
                    user_msg,
                    model_settings={"temperature": 0.8},
                ),
                timeout=60.0
            )
            retry_text = str(result_retry.output).strip() if result_retry.output else ""
            retry_text = _strip_meta_commentary(retry_text)
            if _validate_output_quality(retry_text, clean_draft):
                full_text = retry_text
            else:
                pass
        except Exception:
            pass

    if not full_text:
        return

    # Post-processing pipeline
    text = _apply_post_processing(full_text, style.language, style_mode)
    text = _inject_short_sentences(text, style.language, style_mode)
    text = _programmatic_sentence_humanize(text, style.language, style_mode)
    text = _apply_post_processing(text, style.language, style_mode)
    text = _validate_paragraph_count(text, paragraph_count, clean_draft)
    text = _enforce_min_sentences(text, min_sentences=4)

    # ── Trigram check — Pass 2 jika overlap terlalu tinggi ───
    trigram_overlap = check_trigram_overlap(clean_draft, text)

    if _needs_rewrite(trigram_overlap, threshold=0.30):
        # Overlap terlalu tinggi → minta Groq rewrite lagi
        # dengan instruksi structural yang lebih agresif
        pass2_msg = (
            f"TUGAS: Tulis ulang teks berikut.\n"
            f"ATURAN MUTLAK:\n"
            f"- Output HANYA teks yang sudah diubah\n"
            f"- DILARANG menulis penjelasan, catatan, "
            f"  atau daftar perubahan apapun\n"
            f"- DILARANG menulis kalimat seperti "
            f"  'berikut teks ulang' atau 'perubahan yang dilakukan'\n"
            f"- Langsung mulai dengan kalimat pertama teks\n\n"
            f"FOKUS: Ubah struktur kalimat secara fundamental. "
            f"Tidak boleh ada 3 kata berurutan yang sama. "
            f"Pertahankan {paragraph_count} paragraf.\n\n"
            f"TEKS YANG HARUS DITULIS ULANG:\n{text}"
        )
        try:
            result2 = await asyncio.wait_for(
                agent.run(
                    pass2_msg,
                    model_settings={"temperature": 1.2},
                ),
                timeout=60.0
            )
            text2 = str(result2.output).strip() if result2.output else ""
            text2 = _strip_meta_commentary(text2)
            if text2:
                text2 = _apply_post_processing(text2, style.language, style_mode)
                text2 = _programmatic_sentence_humanize(text2, style.language, style_mode)
                text2 = _validate_paragraph_count(text2, paragraph_count, clean_draft)
                text2 = _enforce_min_sentences(text2, min_sentences=4)
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
        
    system_prompt = _build_system_prompt(style, paragraph_count)
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
        f"PENTING: setiap paragraf HARUS minimal 4 kalimat "
        f"dengan variasi panjang yang ekstrem. "
        f"Ikuti register dan gaya yang telah ditentukan. "
        f"Kembalikan HANYA teks hasil rewrite — "
        f"jangan tambahkan penjelasan, label, atau komentar. "
        f"Output WAJIB persis {paragraph_count} paragraf "
        f"dipisahkan baris kosong.\n\n"
        f"{clean_draft}"
    )

    try:
        result = await asyncio.wait_for(
            agent.run(
                user_msg,
                model_settings={"temperature": 1.2},
            ),
            timeout=60.0
        )
        full_text = str(result.output).strip() if result.output else ""
    except Exception:
        full_text = draft  # fallback ke draf asli

    # Bersihkan meta-commentary sebelum validasi
    full_text = _strip_meta_commentary(full_text)

    # Validasi output Groq sebelum post-processing
    if not _validate_output_quality(full_text, clean_draft):
        # Output tidak layak — retry dengan temperature lebih rendah
        try:
            result_retry = await asyncio.wait_for(
                agent.run(
                    user_msg,
                    model_settings={"temperature": 0.8},
                ),
                timeout=60.0
            )
            retry_text = str(result_retry.output).strip() if result_retry.output else ""
            retry_text = _strip_meta_commentary(retry_text)
            if _validate_output_quality(retry_text, clean_draft):
                full_text = retry_text
            else:
                pass
        except Exception:
            pass

    text = _apply_post_processing(full_text, style.language, style_mode)
    text = _inject_short_sentences(text, style.language, style_mode)
    text = _programmatic_sentence_humanize(text, style.language, style_mode)
    text = _apply_post_processing(text, style.language, style_mode)
    text = _validate_paragraph_count(text, paragraph_count, clean_draft)
    text = _enforce_min_sentences(text, min_sentences=4)

    # ── Trigram check — Pass 2 jika overlap terlalu tinggi ───
    trigram_overlap = check_trigram_overlap(clean_draft, text)

    if _needs_rewrite(trigram_overlap, threshold=0.30):
        pass2_msg = (
            f"TUGAS: Tulis ulang teks berikut.\n"
            f"ATURAN MUTLAK:\n"
            f"- Output HANYA teks yang sudah diubah\n"
            f"- DILARANG menulis penjelasan, catatan, "
            f"  atau daftar perubahan apapun\n"
            f"- DILARANG menulis kalimat seperti "
            f"  'berikut teks ulang' atau 'perubahan yang dilakukan'\n"
            f"- Langsung mulai dengan kalimat pertama teks\n\n"
            f"FOKUS: Ubah struktur kalimat secara fundamental. "
            f"Tidak boleh ada 3 kata berurutan yang sama. "
            f"Pertahankan {paragraph_count} paragraf.\n\n"
            f"TEKS YANG HARUS DITULIS ULANG:\n{text}"
        )
        try:
            result2 = await asyncio.wait_for(
                agent.run(
                    pass2_msg,
                    model_settings={"temperature": 1.2},
                ),
                timeout=60.0
            )
            text2 = str(result2.output).strip() if result2.output else ""
            text2 = _strip_meta_commentary(text2)
            if text2:
                text2 = _apply_post_processing(text2, style.language, style_mode)
                text2 = _programmatic_sentence_humanize(text2, style.language, style_mode)
                text2 = _validate_paragraph_count(text2, paragraph_count, clean_draft)
                text2 = _enforce_min_sentences(text2, min_sentences=4)
                new_overlap = check_trigram_overlap(clean_draft, text2)
                if new_overlap < trigram_overlap:
                    text = text2
                    trigram_overlap = new_overlap
        except Exception:
            pass

    changes = _generate_changes_made(clean_draft, text, trigram_overlap)
    return ProcessedText(final_text=text, changes_made=changes)

import re
import random

def _programmatic_sentence_humanize(text: str, lang: str, style_mode: str = "populer") -> str:
    """Rule-based sentence transformation. No LLM — 100% reliable."""
    if lang not in ("id", "mixed"):
        return text

    if style_mode in ("akademik", "profesional"):
        opener_rules = [
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

    word_rules = [
        (r'\btersebut\b', 'ini'),
        (r'\bmenunjukkan bahwa\b', 'mengindikasikan bahwa'),
        (r'\bdapat disimpulkan\b', 'dapat dikatakan'),
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

            for pattern, replacement in opener_rules:
                new_s = re.sub(pattern, replacement, s, count=1, flags=re.IGNORECASE)
                if new_s != s:
                    s = new_s[0].upper() + new_s[1:] if new_s else new_s
                    break

            for pattern, replacement in word_rules:
                s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)

            if len(s.split()) > 22:
                parts = re.split(r'\bsehingga\b', s, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    p0 = parts[0].strip().rstrip(',')
                    p1 = parts[1].strip()
                    if len(p0.split()) > 5 and len(p1.split()) > 3:
                        p1 = p1[0].upper() + p1[1:]
                        s = p0 + '. ' + p1

            if len(s.split()) > 22:
                parts = re.split(r'\byang mana\b', s, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    p0 = parts[0].strip().rstrip(',')
                    p1 = parts[1].strip()
                    if len(p0.split()) > 5 and len(p1.split()) > 3:
                        p1 = p1[0].upper() + p1[1:]
                        s = p0 + '. ' + p1

            if len(s.split()) > 25:
                parts = re.split(r',?\s+yang\s+', s, maxsplit=1)
                if len(parts) == 2 and len(parts[0].split()) >= 6:
                    p1 = parts[0].strip().rstrip(',') + '.'
                    p2 = parts[1].strip()
                    if p2:
                        p2 = p2[0].upper() + p2[1:]
                        s = p1 + ' ' + p2
            
            if len(s.split()) > 28:
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
            "id": ["Menariknya.", "Perlu dicatat.", "Ini krusial.", "Patut diperhatikan."],
            "en": ["Notably.", "Worth noting.", "This matters.", "Interestingly."]
        },
        "profesional": {
            "id": ["Faktanya.", "Singkat kata.", "Konkretnya.", "Perlu diperhatikan."],
            "en": ["In fact.", "Simply put.", "Concretely.", "Worth noting."]
        },
        "kreatif": {
            "id": ["Diam.", "Sunyi.", "Tiba-tiba.", "Aneh.", "Betul juga."],
            "en": ["Silence.", "Strange.", "Suddenly.", "Indeed."]
        },
        "populer": {
            "id": ["Menarik.", "Sungguh.", "Wajar saja.", "Memang begitu.", "Cukup rumit.", "Tentu.", "Tidak mudah.", "Perlu diakui."],
            "en": ["Interesting.", "Indeed.", "Fair enough.", "That said.", "Not easy.", "Worth noting."]
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
        injected = False
        for sent in sentences:
            new_sentences.append(sent)
            if len(sent.split()) > 18 and random.random() < 0.40:
                new_sentences.append(random.choice(short_list))
                injected = True
        
        if not injected and len(sentences) >= 3:
            if random.random() < 0.5:
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
    
    web_artifacts = [
        r'Enter fullscreen mode\.?',
        r'Exit fullscreen mode\.?',
        r'─+\s*─*',
    ]
    for p in web_artifacts:
        text = re.sub(p, '', text, flags=re.IGNORECASE)
        
    if lang in ("id", "mixed"):
        universal_replacements = [
            (r'(?i)\bini merupakan\b', 'ini adalah'),
            (r'(?i)\byang merupakan\b', 'yang menjadi'),
            (r'(?i)\bmerupakan sebuah\b', 'adalah sebuah'),
            (r'(?i)\bmerupakan salah satu\b', 'menjadi salah satu'),
            (r'(?i)\bmerupakan\b', 'menjadi'),
            (r'(?i)\btidak memiliki\b', 'tidak punya'),
            (r'(?i)\bmemiliki kemampuan\b', 'mampu'),
            (r'(?i)\bmemiliki peran\b', 'berperan'),
            (r'(?i)\bmemiliki dampak\b', 'berdampak'),
            (r'(?i)\bmemiliki\b', 'punya'),
            (r'(?i)\bberbagai macam\b', 'beragam'),
            (r'(?i)\bberbagai\b', 'beragam'),
            (r'(?i)\bsehingga\b', 'hingga'),
            (r'(?i)\bserta\b', 'dan juga'),
            (r'(?i)\bhal tersebut\b', 'hal ini'),
            (r'(?i)\bmasalah tersebut\b', 'masalah ini'),
            (r'(?i)\bprogram tersebut\b', 'program ini'),
            (r'(?i)\bkondisi tersebut\b', 'kondisi itu'),
            (r'(?i)\bsituasi tersebut\b', 'situasi itu'),
            (r'(?i)\bhal ini menunjukkan bahwa\b', 'ini menandakan bahwa'),
            (r'(?i)\bhal ini membuktikan\b', 'fakta ini membuktikan'),
            (r'(?i)\bhal ini\b', 'kondisi ini'),
            (r'(?i)^selain itu,?\s*', 'Di samping itu, '),
            (r'(?i)\bselain itu,?\b', 'di samping itu,'),
            (r'(?i)^oleh karena itu,?\s*', 'Karena itu, '),
            (r'(?i)\boleh karena itu\b', 'karena itu'),
            (r'(?i)^dengan demikian,?\s*', 'Dengan begitu, '),
            (r'(?i)\bdengan demikian\b', 'dengan begitu'),
            (r'(?i)\bdapat disimpulkan bahwa\b', 'intinya,'),
            (r'(?i)\bdapat disimpulkan\b', 'dapat dikatakan'),
            (r'(?i)^secara keseluruhan,?\s*', 'Bila dilihat secara utuh, '),
            (r'(?i)\bsecara keseluruhan\b', 'secara menyeluruh'),
            (r'(?i)\bsangat penting untuk\b', 'mendesak untuk'),
            (r'(?i)\bsangat penting\b', 'amat krusial'),
            (r'(?i)\bperlu dicatat bahwa\b', 'perlu diperhatikan,'),
            (r'(?i)\bperlu dicatat\b', 'patut dicermati'),
            (r'(?i)^dalam hal ini,?\s*', 'Dalam konteks ini, '),
            (r'(?i)\bdalam hal ini\b', 'dalam konteks ini'),
            (r'(?i)^adapun\b,?\s*', 'Sementara itu, '),
            (r'(?i)\badapun\b', 'sementara itu'),
            (r'(?i)\btentunya\b', 'tentu'),
            (r'(?i)\bpastinya\b', 'sudah pasti'),
            (r'(?i)\btidak hanya itu\b', 'lebih dari itu'),
            (r'(?i)^di sisi lain,?\s*', 'Sebaliknya, '),
            (r'(?i)\bdi sisi lain\b', 'sebaliknya'),
            (r'(?i)\btersebut\b', 'itu'),
        ]

        for pattern, replacement in universal_replacements:
            text = _replace_preserve_case(text, pattern, replacement)

        text = re.sub(r',\s*,', ',', text)
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
        
    for pattern, replacement in replacements:
        text = _replace_preserve_case(text, pattern, replacement)
    
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
                if random.random() < 0.08:
                    sent_lower = sentences[i].lower()
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
    
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

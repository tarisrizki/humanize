"""Style Analyzer — Extracts a quantitative StyleProfile from user texts.

Uses spaCy for English NLP, Stanza for Indonesian NLP, textstat for readability,
and langdetect for language detection.
"""

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

import textstat
from langdetect import detect

from app.models.style_profile import StyleProfile


# ── Common data structures (abstraction over spaCy / Stanza) ──────────────────

@dataclass
class Token:
    """Unified token representation."""
    text: str
    lower: str
    pos: str        # Universal POS tag
    dep: str        # Dependency relation
    is_punct: bool
    is_space: bool


@dataclass
class Sentence:
    """Unified sentence representation."""
    text: str
    tokens: list[Token] = field(default_factory=list)


@dataclass
class ParsedDoc:
    """Unified document parsed by either spaCy or Stanza."""
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def tokens(self) -> list[Token]:
        return [t for s in self.sentences for t in s.tokens]

    @property
    def word_tokens(self) -> list[Token]:
        return [t for t in self.tokens if not t.is_punct and not t.is_space]


# ── Constants ─────────────────────────────────────────────────────────────────

TRANSITION_WORDS_EN = {
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "consequently", "meanwhile", "additionally", "nonetheless", "thus",
    "hence", "instead", "otherwise", "likewise", "similarly",
    "in addition", "as a result", "on the other hand", "in contrast",
    "for example", "for instance", "in fact", "of course", "after all",
    "at the same time", "in other words", "on the contrary",
}

TRANSITION_WORDS_ID = {
    "namun", "oleh karena itu", "selain itu", "meskipun demikian",
    "dengan demikian", "sementara itu", "di sisi lain", "sebagai contoh",
    "misalnya", "pada kenyataannya", "justru", "sebaliknya",
    "akibatnya", "kemudian", "akan tetapi", "walaupun", "bahkan",
    "terlebih lagi", "sebab", "karena",
}

HEDGING_WORDS_EN = {
    "perhaps", "maybe", "possibly", "might", "could", "seem",
    "appear", "suggest", "tend", "somewhat", "rather", "quite",
    "fairly", "probably", "likely", "unlikely", "approximately",
    "roughly", "generally", "typically", "usually", "often",
}

HEDGING_WORDS_ID = {
    "mungkin", "barangkali", "sepertinya", "tampaknya", "agaknya",
    "kiranya", "kira-kira", "kurang lebih", "biasanya", "umumnya",
    "seringkali", "kadang-kadang", "nampaknya", "rasanya",
}

FORMAL_MARKERS_EN = {
    "furthermore", "moreover", "consequently", "therefore", "thus",
    "hence", "nevertheless", "notwithstanding", "aforementioned",
    "hitherto", "whereby", "wherein", "thereof",
}

INFORMAL_MARKERS_EN = {
    "gonna", "wanna", "kinda", "sorta", "yeah", "nope", "cool",
    "awesome", "stuff", "thing", "basically", "literally", "like",
    "you know", "i mean", "right", "ok", "okay", "hey", "wow",
}

FORMAL_MARKERS_ID = {
    "demikian", "sebagaimana", "berdasarkan", "sehubungan",
    "memperhatikan", "mengingat", "adapun", "perihal",
    "terkait", "merujuk", "sesuai", "perihal",
}

INFORMAL_MARKERS_ID = {
    "gue", "lo", "gw", "lu", "nih", "dong", "sih", "deh",
    "banget", "emang", "udah", "gimana", "kayak", "bgt",
    "wkwk", "haha", "anjir", "cuy", "bro", "sis",
}


# ── NLP Engine Registry ──────────────────────────────────────────────────────

_spacy_nlp = None        # spaCy English model
_stanza_pipeline = None  # Stanza Indonesian pipeline


def register_spacy(nlp) -> None:
    """Register the loaded spaCy English model."""
    global _spacy_nlp
    _spacy_nlp = nlp


def register_stanza(pipeline) -> None:
    """Register the loaded Stanza Indonesian pipeline."""
    global _stanza_pipeline
    _stanza_pipeline = pipeline


def _parse_with_spacy(text: str) -> ParsedDoc:
    """Parse text using spaCy (English)."""
    if _spacy_nlp is None:
        raise RuntimeError("spaCy model not loaded. Start the server first.")

    nlp = _spacy_nlp
    nlp.max_length = max(nlp.max_length, len(text) + 1000)
    doc = nlp(text)

    sentences = []
    for sent in doc.sents:
        tokens = [
            Token(
                text=t.text,
                lower=t.lower_,
                pos=t.pos_,
                dep=t.dep_,
                is_punct=t.is_punct,
                is_space=t.is_space,
            )
            for t in sent
        ]
        sentences.append(Sentence(text=sent.text, tokens=tokens))

    return ParsedDoc(sentences=sentences)


def _parse_with_stanza(text: str) -> ParsedDoc:
    """Parse text using Stanza (Indonesian)."""
    if _stanza_pipeline is None:
        raise RuntimeError("Stanza Indonesian pipeline not loaded. Start the server first.")

    doc = _stanza_pipeline(text)

    sentences = []
    for sent in doc.sentences:
        tokens = []
        for word in sent.words:
            is_punct = word.upos == "PUNCT"
            is_space = word.text.strip() == ""
            tokens.append(
                Token(
                    text=word.text,
                    lower=word.text.lower(),
                    pos=word.upos,           # Stanza uses Universal POS tags
                    dep=word.deprel or "",    # Dependency relation
                    is_punct=is_punct,
                    is_space=is_space,
                )
            )
        sentences.append(Sentence(text=sent.text, tokens=tokens))

    return ParsedDoc(sentences=sentences)


def _parse_text(text: str, lang: str) -> ParsedDoc:
    """Route text to the appropriate NLP engine based on language."""
    if lang == "id" and _stanza_pipeline is not None:
        return _parse_with_stanza(text)
    elif _spacy_nlp is not None:
        return _parse_with_spacy(text)
    else:
        raise RuntimeError("No NLP model available. Start the server first.")


# ── Language detection ────────────────────────────────────────────────────────

def _detect_language(texts: list[str]) -> str:
    """Detect the dominant language across multiple texts."""
    combined = " ".join(texts)[:5000]
    try:
        lang = detect(combined)
        if lang in ("id", "ms"):
            return "id"
        if lang == "en":
            return "en"
        return "mixed"
    except Exception:
        return "en"


# ── Analysis helpers (work with ParsedDoc) ────────────────────────────────────

def _sentence_lengths(doc: ParsedDoc) -> list[int]:
    """Return word-count per sentence."""
    return [len([t for t in s.tokens if not t.is_punct and not t.is_space])
            for s in doc.sentences]


def _active_voice_ratio(doc: ParsedDoc, lang: str) -> float:
    """Estimate active vs passive voice ratio using dependency parsing."""
    total = len(doc.sentences)
    if total == 0:
        return 1.0

    passive = 0
    for sent in doc.sentences:
        for token in sent.tokens:
            if lang == "en":
                # English: look for auxpass / nsubjpass
                if token.dep in ("auxpass", "nsubjpass"):
                    passive += 1
                    break
            else:
                # Indonesian: passive often marked with "di-" prefix verbs
                # or dependency relation containing "pass"
                if "pass" in token.dep.lower():
                    passive += 1
                    break
                if token.pos == "VERB" and token.lower.startswith("di"):
                    passive += 1
                    break

    return round(1.0 - (passive / total), 4)


def _count_transitions(text_lower: str, lang: str) -> Counter:
    """Count transition word/phrase occurrences."""
    words = TRANSITION_WORDS_EN if lang == "en" else TRANSITION_WORDS_EN | TRANSITION_WORDS_ID
    counter: Counter = Counter()
    for phrase in words:
        count = text_lower.count(phrase)
        if count > 0:
            counter[phrase] = count
    return counter


def _hedging_frequency(text_lower: str, sentence_count: int, lang: str) -> float:
    """Hedging expressions per sentence."""
    words = HEDGING_WORDS_EN if lang == "en" else HEDGING_WORDS_EN | HEDGING_WORDS_ID
    total = sum(1 for w in words if w in text_lower)
    return round(total / max(sentence_count, 1), 3)


def _emotion_register(text_lower: str, lang: str) -> str:
    """Heuristic: formal vs informal vs neutral."""
    if lang == "id":
        formal_count = sum(1 for w in FORMAL_MARKERS_ID if w in text_lower)
        informal_count = sum(1 for w in INFORMAL_MARKERS_ID if w in text_lower)
    else:
        formal_count = sum(1 for w in FORMAL_MARKERS_EN if w in text_lower)
        informal_count = sum(1 for w in INFORMAL_MARKERS_EN if w in text_lower)

    if formal_count > informal_count + 2:
        return "formal"
    if informal_count > formal_count + 2:
        return "informal"
    return "neutral"


def _favorite_phrases(doc: ParsedDoc, top_n: int = 10) -> list[str]:
    """Extract recurring n-grams (2-4 words), excluding stopwords-only phrases."""
    tokens = [t.lower for t in doc.word_tokens]

    STOPWORDS = {
        "the", "a", "an", "is", "are", "in", "of", "to", "and", "for",
        "on", "it", "that", "this", "with", "as", "was", "at", "by", "from",
        # Indonesian stopwords
        "yang", "di", "dan", "ke", "dari", "ini", "itu", "dengan",
        "untuk", "pada", "adalah", "akan", "juga", "sudah", "tidak",
    }

    counter: Counter = Counter()
    for n in (2, 3, 4):
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            if not all(t in STOPWORDS for t in gram):
                counter[" ".join(gram)] += 1

    return [phrase for phrase, count in counter.most_common(top_n) if count >= 2]


def _lexical_diversity(doc: ParsedDoc) -> float:
    """Type-Token Ratio: unique words / total words."""
    tokens = [t.lower for t in doc.word_tokens]
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def _paragraph_length_avg(text: str, doc: ParsedDoc) -> float:
    """Average number of sentences per paragraph."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return 0.0
    sentence_count = len(doc.sentences)
    return round(sentence_count / len(paragraphs), 2)


def _extract_few_shot_examples(texts: list[str], avg_sentence_length: float, max_words: int = 500) -> list[str]:
    """Extract representative paragraphs closest to the average sentence length, up to max_words."""
    valid_paras = []
    
    for text in texts:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for p in paragraphs:
            words = p.split()
            word_count = len(words)
            if 20 <= word_count <= 150:
                sentences = re.split(r'[.!?]+', p)
                sentences = [s for s in sentences if s.strip()]
                if not sentences:
                    continue
                    
                para_avg_sl = word_count / len(sentences)
                distance = abs(para_avg_sl - avg_sentence_length)
                
                valid_paras.append({
                    "text": p,
                    "word_count": word_count,
                    "distance": distance
                })
            
    if not valid_paras:
        return []
        
    # Sort by closest to mean
    valid_paras.sort(key=lambda x: x["distance"])
    
    sampled = []
    current_words = 0
    for item in valid_paras:
        if current_words + item["word_count"] > max_words and sampled:
            break
        sampled.append(item["text"])
        current_words += item["word_count"]
        
    return sampled

# ── Main analysis function ───────────────────────────────────────────────────

def analyze_style(texts: list[str], user_id: str = "anonymous") -> StyleProfile:
    """Analyze a collection of texts and produce a StyleProfile incrementally."""
    if not texts:
        return StyleProfile(user_id=user_id)

    # Detect language using a sample
    lang = _detect_language(texts)

    all_sent_lens = []
    flesch_scores = []
    total_sentences = 0
    total_passive = 0
    transitions_counter = Counter()
    hedge_count = 0
    formal_count = 0
    informal_count = 0
    
    all_word_tokens = []
    unique_tokens = set()
    total_paragraphs = 0

    # Process sequentially to save memory
    for text in texts:
        text = text.strip()
        if not text:
            continue
            
        # Paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        total_paragraphs += len(paragraphs)
        
        # Parse chunk
        # To avoid spaCy/Stanza crashing on huge files, we could split by 100k chars
        # For MVP, assuming individual documents in `texts` are reasonable length.
        try:
            doc = _parse_text(text, lang)
        except Exception:
            # Skip unparseable chunks
            continue
            
        # Sentence lengths
        lens = _sentence_lengths(doc)
        all_sent_lens.extend(lens)
        total_sentences += len(doc.sentences)
        
        # Readability
        try:
            flesch_scores.append(textstat.flesch_reading_ease(text))
        except:
            pass
            
        # Voice
        for sent in doc.sentences:
            for token in sent.tokens:
                if lang == "en":
                    if token.dep in ("auxpass", "nsubjpass"):
                        total_passive += 1
                        break
                else:
                    if "pass" in token.dep.lower():
                        total_passive += 1
                        break
                    if token.pos == "VERB" and token.lower.startswith("di"):
                        total_passive += 1
                        break

        # Text lower for keyword matching
        text_lower = text.lower()
        
        # Transitions
        trans_words = TRANSITION_WORDS_EN if lang == "en" else TRANSITION_WORDS_EN | TRANSITION_WORDS_ID
        for phrase in trans_words:
            count = text_lower.count(phrase)
            if count > 0:
                transitions_counter[phrase] += count
                
        # Hedging
        hedge_words = HEDGING_WORDS_EN if lang == "en" else HEDGING_WORDS_EN | HEDGING_WORDS_ID
        hedge_count += sum(1 for w in hedge_words if w in text_lower)
        
        # Register
        if lang == "id":
            formal_count += sum(1 for w in FORMAL_MARKERS_ID if w in text_lower)
            informal_count += sum(1 for w in INFORMAL_MARKERS_ID if w in text_lower)
        else:
            formal_count += sum(1 for w in FORMAL_MARKERS_EN if w in text_lower)
            informal_count += sum(1 for w in INFORMAL_MARKERS_EN if w in text_lower)
            
        # Vocabulary
        word_toks = [t.lower for t in doc.word_tokens]
        all_word_tokens.extend(word_toks)
        unique_tokens.update(word_toks)

    # --- Aggregation ---
    
    avg_sl = round(statistics.mean(all_sent_lens), 2) if all_sent_lens else 0.0
    std_sl = round(statistics.stdev(all_sent_lens), 2) if len(all_sent_lens) > 1 else 0.0
    
    flesch = round(statistics.mean(flesch_scores), 2) if flesch_scores else 0.0
    
    av_ratio = 1.0
    if total_sentences > 0:
        av_ratio = round(1.0 - (total_passive / total_sentences), 4)
        
    top_trans = [phrase for phrase, _ in transitions_counter.most_common(10)]
    
    hedge_freq = round(hedge_count / max(total_sentences, 1), 3)
    
    register = "neutral"
    if formal_count > informal_count + 2:
        register = "formal"
    elif informal_count > formal_count + 2:
        register = "informal"
        
    # Phrases (Approximate over flattened tokens)
    fav_phrases = []
    if all_word_tokens:
        STOPWORDS = {
            "the", "a", "an", "is", "are", "in", "of", "to", "and", "for",
            "on", "it", "that", "this", "with", "as", "was", "at", "by", "from",
            "yang", "di", "dan", "ke", "dari", "ini", "itu", "dengan",
            "untuk", "pada", "adalah", "akan", "juga", "sudah", "tidak",
        }
        phrase_counter = Counter()
        for n in (2, 3, 4):
            for i in range(len(all_word_tokens) - n + 1):
                gram = tuple(all_word_tokens[i : i + n])
                if not all(t in STOPWORDS for t in gram):
                    phrase_counter[" ".join(gram)] += 1
        fav_phrases = [p for p, c in phrase_counter.most_common(10) if c >= 2]

    lex_div = round(len(unique_tokens) / len(all_word_tokens), 4) if all_word_tokens else 0.0
    para_avg = round(total_sentences / total_paragraphs, 2) if total_paragraphs else 0.0
    
    # Few-shot examples
    few_shot = _extract_few_shot_examples(texts, avg_sl, max_words=500)

    return StyleProfile(
        user_id=user_id,
        language=lang,
        avg_sentence_length=avg_sl,
        sentence_length_std=std_sl,
        flesch_reading_ease=flesch,
        active_voice_ratio=av_ratio,
        top_transitions=top_trans,
        hedging_frequency=hedge_freq,
        emotion_register=register,
        favorite_phrases=fav_phrases,
        lexical_diversity=lex_div,
        paragraph_length_avg=para_avg,
        few_shot_examples=few_shot,
    )

import stanza

# Load once at module level
_pipelines = {}

def _get_pipeline(lang: str):
    if lang not in _pipelines:
        _pipelines[lang] = stanza.Pipeline(
            lang=lang,
            processors="tokenize",  # Just tokenize, no need for NER/POS
            use_gpu=False
        )
    return _pipelines[lang]

def score_humanness(text: str, lang: str = "id") -> dict:
    """
    Calculate local burstiness using Stanza.
    Supports 'id' (Indonesia) and 'en' (English).
    """
    if not text:
        return {"burstiness_score": 0.0, "is_human_like": False, "reason": "empty_text"}
        
    nlp = _get_pipeline(lang)
    doc = nlp(text)

    sent_lengths = [
        len(sent.tokens) 
        for sent in doc.sentences
    ]

    if len(sent_lengths) < 3:
        # Teks terlalu pendek, tidak bisa dievaluasi
        return {"burstiness_score": 0.0, "is_human_like": True, "reason": "too_short"}

    avg = sum(sent_lengths) / len(sent_lengths)
    variance = sum((l - avg) ** 2 for l in sent_lengths) / len(sent_lengths)
    burstiness = variance ** 0.5

    # Distribusi kalimat ekstrem: ada yang sangat pendek DAN sangat panjang
    has_short = any(l <= 5 for l in sent_lengths)
    has_long  = any(l >= 20 for l in sent_lengths)

    return {
        "burstiness_score": round(burstiness, 2),
        "avg_sentence_length": round(avg, 2),
        "sentence_count": len(sent_lengths),
        "has_short_sentences": has_short,
        "has_long_sentences": has_long,
        # Threshold: burstiness >= 3.0 DAN ada variasi ekstrem
        "is_human_like": burstiness >= 3.0 and has_short,
    }

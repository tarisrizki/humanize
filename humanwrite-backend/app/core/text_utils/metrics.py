import re
import statistics

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
    
    delta = len(new_sents) - len(orig_sents)
    if abs(delta) >= 2:
        if delta > 0:
            changes.append(
                "Beberapa kalimat panjang dipecah menjadi kalimat lebih pendek "
                "untuk meningkatkan variasi ritme."
            )
        else:
            changes.append(
                "Beberapa kalimat pendek digabungkan untuk alur yang lebih baik."
            )
    
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

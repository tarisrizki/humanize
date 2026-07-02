import re

def _count_paragraphs(text: str) -> int:
    """Count paragraphs in text (separated by double newlines or single newlines with content)."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n', text) if p.strip()]
    return len(paragraphs)

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

        if len(sentences) >= min_sentences:
            result.append(' '.join(sentences))
            continue

        expanded = list(sentences)
        attempts = 0
        while len(expanded) < min_sentences and attempts < 3:
            longest_idx = max(range(len(expanded)),
                            key=lambda i: len(expanded[i].split()))
            longest = expanded[longest_idx]
            words = longest.split()

            if len(words) < 12:
                break

            mid = len(words) // 2
            split_words = ['yang', 'dan', 'karena', 'sehingga',
                          'namun', 'tetapi', 'meskipun', 'while',
                          'which', 'and', 'but', 'because']

            split_pos = None
            search_start = int(len(words) * 0.35)
            search_end   = int(len(words) * 0.65)
            for i in range(search_start, search_end):
                if words[i].lower() in split_words:
                    split_pos = i
                    break

            if split_pos:
                part1 = ' '.join(words[:split_pos]).rstrip(',') + '.'
                part2_words = words[split_pos:]
                if part2_words:
                    part2_words[0] = part2_words[0].capitalize()
                part2 = ' '.join(part2_words)
                if not part2.endswith(('.', '!', '?')):
                    part2 += '.'
                expanded[longest_idx:longest_idx+1] = [part1, part2]
            else:
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

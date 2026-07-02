import re

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

def _strip_meta_commentary(text: str) -> str:
    """
    Hapus meta-commentary yang sering ditambahkan Groq:
    - Paragraf penjelasan di awal (sebelum teks asli)
    - Bullet point daftar perubahan di akhir
    - Kalimat transisi model tentang apa yang dilakukannya
    """
    lines = text.strip().split('\n')
    
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
        
        if re.match(r'^perubahan', stripped) or \
           re.match(r'^catatan', stripped) or \
           re.match(r'^note:', stripped):
            skip_bullet_section = True
            continue
        
        if skip_bullet_section and (
            line.strip().startswith('*') or 
            line.strip().startswith('-') or
            line.strip().startswith('•')
        ):
            continue
        
        if skip_bullet_section and len(line.strip()) > 50 and \
           not line.strip().startswith(('*', '-', '•')):
            skip_bullet_section = False
        
        is_meta = any(
            re.match(p, stripped) 
            for p in meta_patterns
        )
        
        if not is_meta:
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    return result if result else text

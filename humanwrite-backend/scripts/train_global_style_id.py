"""Train global style profile using ONLY the clean Indonesian corpus."""

import os
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import re
from app.config import settings
from app.core.style_analyzer import analyze_style, register_spacy, register_stanza
from app.storage.json_store import save_json


def clean_web_artifacts(text: str) -> str:
    """Aggressively clean web artifacts from text."""
    patterns = [
        r'Enter fullscreen mode',
        r'Exit fullscreen mode',
        r'─+',
        r'•',
        r'\[edit\]',
        r'\[\d+\]',
        r'Click here.*',
        r'Read more.*',
        r'Share this.*',
    ]
    for p in patterns:
        text = re.sub(p, '', text, flags=re.IGNORECASE)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_text(file_path: Path) -> str:
    """Extract and clean text from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
        
        # Strip HTML if present
        if "<html" in raw.lower() or "<body" in raw.lower():
            from bs4 import BeautifulSoup
            raw = BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)
        
        return clean_web_artifacts(raw)
    except Exception as e:
        print(f"  Warning: Failed to read {file_path.name}: {e}")
        return ""


def load_nlp_models():
    """Load spaCy and Stanza models."""
    print("Loading NLP models...")
    
    import spacy
    try:
        nlp_en = spacy.load(settings.SPACY_MODEL_EN)
    except OSError:
        spacy.cli.download(settings.SPACY_MODEL_EN)
        nlp_en = spacy.load(settings.SPACY_MODEL_EN)
    register_spacy(nlp_en)

    import stanza
    stanza.download("id", verbose=False)
    nlp_id = stanza.Pipeline("id", processors="tokenize,pos,lemma,depparse", verbose=False)
    register_stanza(nlp_id)
    print("Models loaded.")


def main():
    # Use the Indonesian corpus directory
    corpus_dir = backend_dir / "data" / "training_corpus_informal"
    
    print(f"Reading texts from {corpus_dir}...")
    if not corpus_dir.exists():
        print(f"ERROR: {corpus_dir} does not exist.")
        print("Run download_id_corpus.py first!")
        return
    
    files = list(corpus_dir.glob("*.txt"))
    
    if not files:
        print(f"No .txt files found in {corpus_dir}")
        return
    
    print(f"Found {len(files)} Indonesian corpus files.")
    
    texts = []
    for f in files:
        t = extract_text(f)
        if t and len(t) > 100:
            texts.append(t)
            print(f"  Loaded: {f.name} ({len(t)} chars)")
    
    if not texts:
        print("No valid texts extracted. Exiting.")
        return
    
    print(f"\nTotal texts to analyze: {len(texts)}")
    
    load_nlp_models()
    
    print("Analyzing Indonesian style profile...")
    profile = analyze_style(texts, user_id="global")
    
    # Clean up the profile - remove any web artifacts from favorite phrases
    clean_phrases = [
        p for p in profile.favorite_phrases
        if not any(bad in p.lower() for bad in [
            "fullscreen", "enter ", "exit ", "mode", "─",
            "click", "read more", "subscribe", "share"
        ])
    ]
    profile.favorite_phrases = clean_phrases[:10]
    
    # Clean up few-shot examples - remove any that contain web artifacts
    clean_examples = []
    for ex in profile.few_shot_examples:
        if not any(bad in ex.lower() for bad in [
            "fullscreen", "enter fullscreen", "exit fullscreen",
            "─", "•", "click here"
        ]):
            clean_examples.append(ex)
    profile.few_shot_examples = clean_examples[:6]
    
    # Save
    profiles_dir = backend_dir / "data" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = profiles_dir / "global_style.json"
    save_json(out_path, profile.model_dump())
    
    print(f"\n{'=' * 60}")
    print(f"Global style profile saved to: {out_path}")
    print(f"Language: {profile.language}")
    print(f"Avg sentence length: {profile.avg_sentence_length}")
    print(f"Sentence length std: {profile.sentence_length_std}")
    print(f"Active voice ratio: {profile.active_voice_ratio:.0%}")
    print(f"Favorite phrases: {profile.favorite_phrases}")
    print(f"Few-shot examples: {len(profile.few_shot_examples)} examples")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

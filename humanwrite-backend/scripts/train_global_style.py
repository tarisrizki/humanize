"""Offline script to train the global HumanWrite AI style profile."""

import os
import sys
from pathlib import Path

# Add backend directory to Python path to allow imports from app
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import docx
import pypdf
from app.config import settings
from app.core.style_analyzer import analyze_style, register_spacy, register_stanza
from app.storage.json_store import save_json


def extract_text(file_path: Path) -> str:
    """Extract text from supported file types (.txt, .docx, .pdf)."""
    ext = file_path.suffix.lower()
    text = ""
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            # If the text contains HTML (like our Wikipedia dumps), strip it
            if "<html" in raw_text.lower() or "<body" in raw_text.lower():
                from bs4 import BeautifulSoup
                text = BeautifulSoup(raw_text, "html.parser").get_text(separator=" ", strip=True)
            else:
                text = raw_text
        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)
        elif ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                text = "\n".join(
                    page.extract_text() for page in reader.pages if page.extract_text()
                )
    except Exception as e:
        print(f"Warning: Failed to read {file_path.name}: {e}")
    return text


def load_nlp_models():
    """Load spaCy and Stanza models."""
    print("Loading NLP models (this may take a moment)...")
    
    # Load spaCy
    import spacy
    try:
        nlp_en = spacy.load(settings.SPACY_MODEL_EN)
    except OSError:
        spacy.cli.download(settings.SPACY_MODEL_EN)
        nlp_en = spacy.load(settings.SPACY_MODEL_EN)
    register_spacy(nlp_en)

    # Load Stanza
    import stanza
    stanza.download("id", verbose=False)
    nlp_id = stanza.Pipeline("id", processors="tokenize,pos,lemma,depparse", verbose=False)
    register_stanza(nlp_id)
    print("Models loaded.")


def main():
    corpus_dir = backend_dir / "data" / "training_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all supported files
    files = []
    for ext in ("*.txt", "*.docx", "*.pdf"):
        files.extend(corpus_dir.rglob(ext))
        
    if not files:
        print(f"No documents found in {corpus_dir}. Please add some texts and run again.")
        # We can still create a default style profile for testing purposes
        print("Generating a default global style profile for testing...")
        texts = ["This is a default training text. It has short sentences."]
    else:
        print(f"Found {len(files)} files. Extracting text...")
        texts = []
        for file in files:
            t = extract_text(file)
            if t.strip():
                texts.append(t)
                
    if not texts:
        print("No valid text extracted. Exiting.")
        return

    load_nlp_models()
    
    print("Analyzing global style. This might take a while depending on corpus size...")
    profile = analyze_style(texts, user_id="global")
    
    profiles_dir = backend_dir / "data" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = profiles_dir / "global_style.json"
    save_json(out_path, profile.model_dump())
    
    print(f"Global style profile successfully saved to {out_path}!")


if __name__ == "__main__":
    main()

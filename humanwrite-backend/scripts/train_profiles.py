import os
import sys
from pathlib import Path
import json

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Bypass AppData permissions
os.environ['STANZA_RESOURCES_DIR'] = str(backend_dir / '.stanza_resources')

from app.core.style_analyzer import analyze_style, register_stanza
from app.storage.json_store import save_json

def extract_texts(directory_path):
    if not os.path.exists(directory_path):
        return []
    texts = []
    files = list(Path(directory_path).glob("*.txt"))
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                t = file.read().strip()
                if len(t) > 50:
                    texts.append(t)
        except Exception:
            pass
    return texts

def train_mode(mode_name, corpus_path):
    print(f"--- Training profile for mode: {mode_name} ---")
    texts = extract_texts(corpus_path)
    if not texts:
        print(f"No valid texts found for {mode_name} in {corpus_path}")
        return
        
    print(f"Extracted {len(texts)} files. Analyzing...")
    profile = analyze_style(texts, user_id=f"mode_{mode_name}")
    
    # Clean up few-shot examples
    clean_examples = [ex for ex in profile.few_shot_examples if len(ex.split()) > 10][:6]
    profile.few_shot_examples = clean_examples
    
    out_path = backend_dir / "data" / "profiles" / f"{mode_name}_style.json"
    os.makedirs(out_path.parent, exist_ok=True)
    save_json(out_path, profile.model_dump())
    print(f"Saved {out_path.name} with {len(profile.few_shot_examples)} few-shot examples.\n")


def main():
    print("Loading NLP models...")
    import stanza
    nlp_id = stanza.Pipeline("id", processors="tokenize,pos,lemma,depparse", verbose=False)
    register_stanza(nlp_id)
    print("NLP Models loaded.\n")
    
    # Train specific modes ONLY (this is super fast)
    modes = ["akademik", "profesional", "populer", "kreatif"]
    for mode in modes:
        corpus_path = backend_dir / "data" / f"corpus_{mode}"
        train_mode(mode, corpus_path)

if __name__ == '__main__':
    main()

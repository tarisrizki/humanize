import os
import sys
import pandas as pd
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Bypass AppData permissions
os.environ['STANZA_RESOURCES_DIR'] = str(backend_dir / '.stanza_resources')

from app.core.style_analyzer import analyze_style, register_stanza
from app.storage.json_store import save_json

def extract_texts_by_lang(csv_path):
    if not os.path.exists(csv_path):
        return [], []
    try:
        df = pd.read_csv(csv_path)
        texts_id = df[df["lang"] == "id"]["text"].dropna().tolist()
        texts_en = df[df["lang"] == "en"]["text"].dropna().tolist()
        
        # Take a random sample if it's too huge to process quickly in Stanza, say max 500 per lang
        import random
        random.seed(42)
        if len(texts_id) > 500: texts_id = random.sample(texts_id, 500)
        if len(texts_en) > 500: texts_en = random.sample(texts_en, 500)
            
        return texts_id, texts_en
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return [], []

def train_mode(mode_name, csv_path):
    print(f"--- Training profile for mode: {mode_name} ---")
    texts_id, texts_en = extract_texts_by_lang(csv_path)
    if not texts_id and not texts_en:
        print(f"No valid texts found for {mode_name} in {csv_path}")
        return
        
    print(f"Extracted {len(texts_id)} ID files and {len(texts_en)} EN files. Analyzing...")
    profile = analyze_style(texts_id, texts_en, user_id=f"mode_{mode_name}")
    profile.style_mode = mode_name # Make sure style mode is set correctly in JSON
    
    # Clean up few-shot examples
    profile.few_shot_examples_id = [ex for ex in profile.few_shot_examples_id if len(ex.split()) > 10][:6]
    profile.few_shot_examples_en = [ex for ex in profile.few_shot_examples_en if len(ex.split()) > 10][:6]
    
    out_path = backend_dir / "data" / "profiles" / f"{mode_name}_style.json"
    os.makedirs(out_path.parent, exist_ok=True)
    save_json(out_path, profile.model_dump())
    print(f"Saved {out_path.name} with {len(profile.few_shot_examples_id)} ID and {len(profile.few_shot_examples_en)} EN few-shot examples.\n")


def main():
    print("Loading NLP models...")
    import stanza
    nlp_id = stanza.Pipeline("id", processors="tokenize,pos,lemma,depparse", verbose=False)
    register_stanza(nlp_id)
    print("NLP Models loaded.\n")
    
    modes = ["akademik", "profesional", "populer", "kreatif"]
    for mode in modes:
        csv_path = backend_dir / "data" / f"corpus_{mode}.csv"
        train_mode(mode, csv_path)

if __name__ == '__main__':
    main()

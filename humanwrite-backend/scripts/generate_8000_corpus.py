import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")

backend_dir = Path(__file__).resolve().parent.parent
data_dir = backend_dir / "data"
os.makedirs(data_dir, exist_ok=True)

import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    # Capitalize the first letter of every sentence
    text = re.sub(r'(^[a-z]|\.\s+[a-z])', lambda m: m.group().upper(), text)
    return text

def fetch_dataset(hf_path, hf_name, split, text_field, limit=1000, skip=0, lang="id", trust_remote_code=True, is_translation=False, trans_lang="en"):
    print(f"  Downloading {limit} from {hf_path} ({lang}) skipping {skip}...")
    results = []
    skipped = 0
    try:
        ds = load_dataset(hf_path, hf_name, split=split, streaming=True, trust_remote_code=trust_remote_code)
        for item in ds:
            if skipped < skip:
                skipped += 1
                continue
                
            if is_translation:
                raw = item.get("translation", {}).get(trans_lang, "")
            else:
                raw = item.get(text_field, item.get('content', item.get('article', item.get('cerpen', ''))))
                
            text = clean_text(raw)
            words = text.split()
            if len(words) >= 30:
                if len(words) > 400:
                    # Truncate to ~400 words and add ellipsis
                    text = " ".join(words[:400]) + "..."
                results.append({"text": text, "lang": lang, "source": hf_path})
            if len(results) >= limit:
                break
        print(f"  -> Got {len(results)} items.")
    except Exception as e:
        print(f"  -> Error fetching {hf_path}: {e}")
    return results

def create_csv_for_mode(mode, datasets_config):
    print(f"Processing mode: {mode}")
    all_data = []
    for cfg in datasets_config:
        all_data.extend(fetch_dataset(**cfg))
    
    df = pd.DataFrame(all_data)
    df.index.name = "id"
    out_path = data_dir / f"corpus_{mode}.csv"
    df.to_csv(out_path, index=True)
    print(f"Saved {len(df)} total items to {out_path}\n")

def main():
    print("Starting 8,000 files bilingual dataset generation...")
    
    # 1. Akademik
    create_csv_for_mode("akademik", [
        {"hf_path": "scientific_papers", "hf_name": "pubmed", "split": "train", "text_field": "abstract", "lang": "en"},
        {"hf_path": "indonesian-nlp/wikipedia-id", "hf_name": "default", "split": "train", "text_field": "text", "lang": "id", "trust_remote_code": False}
    ])

    # 2. Profesional
    create_csv_for_mode("profesional", [
        {"hf_path": "SetFit/ag_news", "hf_name": "default", "split": "train", "text_field": "text", "lang": "en"},
        {"hf_path": "hanifabdlh/context-1-indonesian-news-corpus", "hf_name": "default", "split": "train", "text_field": "context", "lang": "id"}
    ])

    # 3. Populer
    create_csv_for_mode("populer", [
        {"hf_path": "abisee/cnn_dailymail", "hf_name": "3.0.0", "split": "train", "text_field": "article", "lang": "en"},
        {"hf_path": "indonesian-nlp/wikipedia-id", "hf_name": "default", "split": "train", "text_field": "text", "lang": "id", "skip": 1000, "trust_remote_code": False}
    ])

    # 4. Kreatif
    create_csv_for_mode("kreatif", [
        {"hf_path": "Helsinki-NLP/opus_books", "hf_name": "en-it", "split": "train", "text_field": "translation", "lang": "en", "is_translation": True, "trans_lang": "en"},
        {"hf_path": "nizzyhussle/cerpen", "hf_name": "default", "split": "train", "text_field": "content", "lang": "id"}
    ])

if __name__ == "__main__":
    main()

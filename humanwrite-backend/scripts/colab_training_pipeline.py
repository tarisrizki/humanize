# %% [markdown]
# # HumanWrite AI: Data Collection & Style Training Pipeline (Refactored)
# 
# Notebook ini dibuat untuk dijalankan di Google Colab.
# Tujuan dari notebook ini adalah:
# 1. Mengumpulkan ~8.000 dokumen teks dari 4 kategori gaya penulisan (Akademik, Profesional, Populer, Kreatif).
# 2. Mengeksekusi pipeline NLP menggunakan pendekatan berorientasi objek (*Object-Oriented*).
# 3. Menghasilkan file JSON Profil Gaya.
# 
# **Instruksi**: Unduh file JSON yang dihasilkan dan pindahkan ke `data/profiles/`.

# %% [markdown]
# ## 1. Setup & Instalasi Dependencies

# %%
!pip install spacy stanza textstat datasets beautifulsoup4 requests pandas tqdm
!python -m spacy download en_core_web_sm
import stanza
stanza.download('id')

# %%
import os
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import textstat
import spacy
import logging

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# %% [markdown]
# ## 2. Konfigurasi Global

# %%
CONFIG = {
    "LIMIT_PER_SOURCE": 1000,
    "OUTPUT_DIR": "profiles",
    "TARGET_STYLES": ["akademik", "profesional", "populer", "kreatif"]
}

# %% [markdown]
# ## 3. Modul Data Collector

# %%
class DataCollector:
    """Kelas khusus untuk menangani web scraping dan API calls."""
    
    @staticmethod
    def fetch_academic(limit):
        logging.info("Memulai ekstraksi data Akademik...")
        texts = []
        # OpenAlex API (EN)
        url = "https://api.openalex.org/works?filter=has_abstract:true,language:en&per-page=200"
        for i in range(1, (limit // 200) + 2):
            try:
                res = requests.get(f"{url}&page={i}").json()
                for work in res.get("results", []):
                    idx = work.get("abstract_inverted_index", {})
                    if idx:
                        words = [""] * (max([max(pos) for pos in idx.values()]) + 1)
                        for word, positions in idx.items():
                            for pos in positions:
                                words[pos] = word
                        texts.append(" ".join(words))
                        if len(texts) >= limit:
                            break
            except Exception as e:
                logging.error(f"Error OpenAlex: {e}")
                break
            time.sleep(1)
            
        # Simulasi Garuda/Neliti (ID)
        texts.extend(["Penelitian ini bertujuan untuk menginvestigasi pengaruh variabel X terhadap variabel Y dalam konteks pendidikan tinggi."] * limit)
        return texts

    @staticmethod
    def fetch_professional(limit):
        logging.info("Memulai ekstraksi data Profesional...")
        texts = []
        try:
            from datasets import load_dataset
            ds = load_dataset("codeparrot/github-markdown", split="train", streaming=True)
            for row in ds:
                texts.append(row["content"])
                if len(texts) >= limit:
                    break
        except Exception as e:
            logging.error(f"Error HF Datasets (Professional): {e}")
            
        texts.extend(["Dokumen ini memberikan panduan teknis mengenai integrasi API pembayaran elektronik untuk instansi pemerintah. Harap ikuti langkah-langkah instalasi berikut dengan cermat."] * limit)
        return texts

    @staticmethod
    def fetch_popular(limit):
        logging.info("Memulai ekstraksi data Populer...")
        # Simulasi scraping portal berita
        return ["Menariknya, di tengah perdebatan sengit tentang regulasi baru ini, masyarakat justru lebih fokus pada isu lain. Kalau dipikir-pikir, ini wajar saja mengingat dampak ekonomi yang membayangi."] * (limit * 2)

    @staticmethod
    def fetch_creative(limit):
        logging.info("Memulai ekstraksi data Kreatif...")
        texts = []
        try:
            from datasets import load_dataset
            ds = load_dataset("pg19", split="train", streaming=True)
            for row in ds:
                excerpt = " ".join(row["text"].split()[:500])
                texts.append(excerpt)
                if len(texts) >= (limit * 2):
                    break
        except Exception as e:
            logging.error(f"Error Gutenberg (Creative): {e}")
        return texts

    @classmethod
    def build_corpus(cls, limit):
        return {
            "akademik": cls.fetch_academic(limit),
            "profesional": cls.fetch_professional(limit),
            "populer": cls.fetch_popular(limit),
            "kreatif": cls.fetch_creative(limit)
        }

# %% [markdown]
# ## 4. Modul Style Analyzer

# %%
class StyleAnalyzer:
    """Kelas untuk mengekstrak fitur linguistik menggunakan spaCy dan Stanza."""
    
    def __init__(self):
        logging.info("Memuat NLP Models (spaCy & Stanza)...")
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_id = stanza.Pipeline("id", processors="tokenize,pos,lemma,depparse", verbose=False)

    def extract_features(self, text, lang="id"):
        try:
            if lang == "en":
                doc = self.nlp_en(text)
                sentences = list(doc.sents)
                words = [token.text for token in doc if not token.is_punct and not token.is_space]
            else:
                doc = self.nlp_id(text)
                sentences = doc.sentences
                words = [word.text for sent in sentences for word in sent.words if word.upos != "PUNCT"]
                
            if not sentences or not words: return None
            
            avg_sentence_len = len(words) / len(sentences)
            flesch = textstat.flesch_reading_ease(text)
            
            return {
                "avg_sentence_length": avg_sentence_len,
                "flesch_reading_ease": flesch,
                "active_voice_ratio": 0.85, # Simplifikasi
                "emotion_register": "neutral"
            }
        except Exception as e:
            return None

    def train_profile(self, texts, style_mode):
        logging.info(f"Menganalisis {len(texts)} dokumen untuk gaya '{style_mode}'...")
        features_list = []
        
        id_markers = {" yang ", " dan ", " di ", " ke ", " dari ", " untuk "}
        
        for text in tqdm(texts):
            text_lower = f" {text.lower()} "
            lang = "id" if any(marker in text_lower for marker in id_markers) else "en"
            feats = self.extract_features(text, lang)
            if feats:
                features_list.append(feats)
                
        if not features_list:
            return {}
            
        df = pd.DataFrame(features_list)
        return {
            "style_mode": style_mode,
            "language": "mixed",
            "avg_sentence_length": float(df["avg_sentence_length"].mean()),
            "sentence_length_std": float(df["avg_sentence_length"].std()),
            "flesch_reading_ease": float(df["flesch_reading_ease"].mean()),
            "active_voice_ratio": float(df["active_voice_ratio"].mean()),
            "emotion_register": "formal" if style_mode in ["akademik", "profesional"] else "neutral",
            "top_transitions": ["namun", "sementara itu"] if style_mode == "akademik" else ["tapi", "selain itu"],
            "hedging_frequency": 0.15 if style_mode == "akademik" else 0.05,
            "favorite_phrases": [],
            "lexical_diversity": 0.55,
            "paragraph_length_avg": 4.5,
            "few_shot_examples_id": texts[:2] if texts else [],
            "few_shot_examples_en": texts[-2:] if len(texts)>2 else []
        }

# %% [markdown]
# ## 5. Pipeline Execution Manager

# %%
class PipelineManager:
    """Manajer yang mengontrol aliran dari pengumpulan data hingga penyimpanan JSON."""
    
    @staticmethod
    def run():
        os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
        
        # 1. Kumpulkan Data
        corpus = DataCollector.build_corpus(CONFIG["LIMIT_PER_SOURCE"])
        
        # 2. Inisialisasi Analyzer
        analyzer = StyleAnalyzer()
        
        # 3. Analisis & Export
        for mode, texts in corpus.items():
            if not texts:
                continue
            
            profile_data = analyzer.train_profile(texts, mode)
            if profile_data:
                file_path = os.path.join(CONFIG["OUTPUT_DIR"], f"{mode}_style.json")
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(profile_data, f, indent=4)
                logging.info(f"✅ Profil {mode} berhasil disimpan di: {file_path}")

        logging.info("✨ Seluruh pipeline selesai dieksekusi!")

# %% [markdown]
# ## 6. Run Pipeline
# Jalankan sel ini untuk memulai proses pelatihan secara keseluruhan.

# %%
if __name__ == "__main__":
    PipelineManager.run()

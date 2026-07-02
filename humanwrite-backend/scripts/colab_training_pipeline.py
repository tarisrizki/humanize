# %% [markdown]
# # HumanWrite AI: Data Collection & Style Training Pipeline (Refactored)
# 
# Notebook ini dibuat untuk dijalankan di Google Colab.
# Tujuan dari notebook ini adalah:
# 1. Mengumpulkan dokumen teks dari 4 kategori gaya penulisan, masing-masing Bahasa Indonesia & Inggris (Total 8 pasang).
# 2. Mengeksekusi pipeline NLP menggunakan pendekatan berorientasi objek (*Object-Oriented*).
# 3. Menghasilkan file JSON Profil Gaya.
# 
# **Instruksi**: Unduh file JSON yang dihasilkan dan pindahkan ke `data/profiles/`.

# %% [markdown]
# ## 1. Setup & Instalasi Dependencies

# %%
!pip install spacy[cuda12x] stanza textstat datasets beautifulsoup4 requests pandas tqdm lxml
!python -m spacy download en_core_web_sm
import stanza
stanza.download('id')

# %%
import os
import json
import requests
import hashlib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import concurrent.futures
import gc
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
    "LIMIT_PER_STYLE": 2000,
    "OUTPUT_DIR": "profiles",
    "CHECKPOINT_DIR": "checkpoints",
    "TARGET_STYLES": [
        "akademik_id", "akademik_en", 
        "profesional_id", "profesional_en", 
        "populer_id", "populer_en", 
        "kreatif_id", "kreatif_en"
    ]
}

if not os.path.exists(CONFIG["OUTPUT_DIR"]):
    os.makedirs(CONFIG["OUTPUT_DIR"])
if not os.path.exists(CONFIG["CHECKPOINT_DIR"]):
    os.makedirs(CONFIG["CHECKPOINT_DIR"])

# %% [markdown]
# ## 3. Modul Data Collector

# %%
class DataCollector:
    """Kelas khusus untuk menangani web scraping dan API calls dari berbagai sumber."""

    @classmethod
    def get_session(cls):
        if not hasattr(cls, "_session"):
            cls._session = requests.Session()
            retry = Retry(connect=3, backoff_factor=0.5, status_forcelist=[ 429, 500, 502, 503, 504 ])
            adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
            cls._session.mount('http://', adapter)
            cls._session.mount('https://', adapter)
        return cls._session

    @staticmethod
    def _fetch_openalex(limit, language="id", sources=None):
        texts = []
        url = f"https://api.openalex.org/works?filter=has_abstract:true,language:{language}"
        if sources:
            url += f",primary_location.source.id:{'|'.join(sources)}"
        
        url += "&per-page=100"
        
        def fetch_page(page):
            headers = {"User-Agent": "mailto:humanwrite.bot@gmail.com"}
            try:
                session = DataCollector.get_session()
                r = session.get(f"{url}&page={page}", headers=headers, timeout=15).json()
                page_texts = []
                for work in r.get("results", []):
                    idx = work.get("abstract_inverted_index", {})
                    if idx:
                        words = [""] * (max([max(pos) for pos in idx.values()]) + 1)
                        for word, positions in idx.items():
                            for pos in positions:
                                words[pos] = word
                        text = " ".join(words)
                        if len(text.split()) > 50:
                            page_texts.append(text)
                return page_texts
            except Exception as e:
                logging.error(f"Error fetching OpenAlex page {page}: {e}")
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_page, i) for i in range(1, 51)]
            for future in concurrent.futures.as_completed(futures):
                for res_text in future.result():
                    if res_text:
                        texts.append(res_text)
                if len(texts) >= limit:
                    for f in futures:
                        f.cancel()
                    break
        return texts[:limit]

    @staticmethod
    def _scrape_rss_feed(feed_url, limit):
        texts = []
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            session = DataCollector.get_session()
            res = session.get(feed_url, headers=headers, timeout=15)
            soup = BeautifulSoup(res.content, "xml")
            
            links = []
            for item in soup.find_all(["item", "entry"]):
                href = None
                for link_tag in item.find_all("link"):
                    if link_tag.get("rel") == ["alternate"] or link_tag.get("rel") == "alternate" or not link_tag.get("rel"):
                        href = link_tag.get("href") or link_tag.text
                        if href and href.strip().startswith("http"):
                            break
                            
                if not href and item.find("link"):
                    link_tag = item.find("link")
                    href = link_tag.get("href") or link_tag.text
                    
                if href and href.strip().startswith("http"):
                    links.append(href.strip())
                
            def fetch_article(link):
                try:
                    session = DataCollector.get_session()
                    html = session.get(link, headers=headers, timeout=15).text
                    soup_art = BeautifulSoup(html, "html.parser")
                    paragraphs = soup_art.find_all("p")
                    content = " ".join([p.text.strip() for p in paragraphs if len(p.text.split()) > 10])
                    return content if content else None
                except Exception as e:
                    logging.error(f"Error fetching article {link}: {e}")
                    return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_article, link) for link in links[:limit]]
                for future in concurrent.futures.as_completed(futures):
                    res_text = future.result()
                    if res_text:
                        texts.append(res_text)
        except Exception as e:
            logging.error(f"Error scraping RSS {feed_url}: {e}")
        return texts

    @staticmethod
    def _fetch_hf_api(dataset, config, split, limit):
        texts = []
        try:
            session = DataCollector.get_session()
            for offset in range(0, limit * 3, 100):
                url = f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length=100"
                res = session.get(url, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    rows = data.get("rows", [])
                    if not rows: break
                    for r in rows:
                        row_data = r.get("row", {})
                        longest_str = ""
                        for val in row_data.values():
                            if isinstance(val, str) and len(val) > len(longest_str):
                                longest_str = val
                        
                        if len(longest_str.split()) > 50:
                            texts.append(longest_str[:3000].rsplit(' ', 1)[0])
                        
                        if len(texts) >= limit: break
                else:
                    break
        except Exception as e:
            logging.error(f"Error HF API {dataset}: {e}")
        return texts[:limit]

    @staticmethod
    def fetch_academic_id(limit):
        return DataCollector._fetch_openalex(limit, "id")

    @staticmethod
    def fetch_academic_en(limit):
        return DataCollector._fetch_openalex(limit, "en")

    @staticmethod
    def fetch_professional_id(limit):
        feeds = [
            "https://www.cnnindonesia.com/nasional/rss",
            "https://www.cnnindonesia.com/ekonomi/rss",
            "https://www.antaranews.com/rss/terkini.xml",
            "https://www.antaranews.com/rss/ekonomi.xml"
        ]
        texts = []
        limit_per_feed = (limit // len(feeds)) + 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(DataCollector._scrape_rss_feed, feed, limit_per_feed) for feed in feeds]
            for future in concurrent.futures.as_completed(futures):
                texts.extend(future.result())
        return texts[:limit]

    @staticmethod
    def fetch_professional_en(limit):
        return DataCollector._fetch_hf_api("abisee/cnn_dailymail", "3.0.0", "train", limit)

    @staticmethod
    def fetch_popular_id(limit):
        feeds = [
            "https://theconversation.com/id/articles.atom",      
            "https://feeds.bbci.co.uk/indonesia/rss.xml",
            "https://www.cnnindonesia.com/gaya-hidup/rss",
            "https://www.cnnindonesia.com/hiburan/rss"
        ]
        texts = []
        limit_per_feed = (limit // len(feeds)) + 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(DataCollector._scrape_rss_feed, feed, limit_per_feed) for feed in feeds]
            for future in concurrent.futures.as_completed(futures):
                texts.extend(future.result())
        return texts[:limit]

    @staticmethod
    def fetch_popular_en(limit):
        feeds = [
            "https://theconversation.com/global/articles.atom",  
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
            "https://www.wired.com/feed/rss"
        ]
        texts = []
        limit_per_feed = (limit // len(feeds)) + 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(DataCollector._scrape_rss_feed, feed, limit_per_feed) for feed in feeds]
            for future in concurrent.futures.as_completed(futures):
                texts.extend(future.result())
        return texts[:limit]

    @staticmethod
    def fetch_creative_id(limit):
        texts_id = DataCollector._fetch_hf_api("allenai/c4", "id", "train", limit)
        if not texts_id:
             texts_id = DataCollector._scrape_rss_feed("https://www.vice.com/id/rss", limit)
        return texts_id[:limit]

    @staticmethod
    def fetch_creative_en(limit):
        gutenberg_urls = [
            "https://www.gutenberg.org/files/1342/1342-0.txt", 
            "https://www.gutenberg.org/files/11/11-0.txt",     
            "https://www.gutenberg.org/files/84/84-0.txt",     
            "https://www.gutenberg.org/files/2701/2701-0.txt"  
        ]
        
        def fetch_gutenberg(url):
            try:
                session = DataCollector.get_session()
                res = session.get(url, timeout=15)
                res.encoding = 'utf-8'
                paragraphs = res.text.split("\n\n")
                valid_paragraphs = [p.replace("\n", " ").strip() for p in paragraphs if len(p.split()) > 100]
                return valid_paragraphs[: (limit // 4)]
            except Exception as e:
                logging.error(f"Error fetching Gutenberg {url}: {e}")
                return []
                
        texts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(fetch_gutenberg, url) for url in gutenberg_urls]
            for future in concurrent.futures.as_completed(futures):
                texts.extend(future.result())
        return texts[:limit]

    @classmethod
    def build_corpus(cls, limit):
        corpus = {}
        logging.info("🚀 Memulai Hyper-Parallel Data Collection untuk semua kategori (Bahasa Terpisah)...")
        
        seen_hashes = set()
        
        def process_and_save(future, name, style_key):
            try:
                texts = future.result()
                deduped = []
                for t in texts:
                    if not t: continue
                    t_hash = hashlib.md5(t.encode('utf-8')).hexdigest()
                    if t_hash not in seen_hashes:
                        seen_hashes.add(t_hash)
                        deduped.append(t)
                
                final_texts = deduped[:limit]
                checkpoint_path = os.path.join(CONFIG["CHECKPOINT_DIR"], f"checkpoint_{style_key}.json")
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump({"count": len(final_texts), "data": final_texts}, f, ensure_ascii=False, indent=2)
                logging.info(f"Checkpoint {name} saved ({len(final_texts)} docs).")
                
                return final_texts
            except Exception as e:
                logging.error(f"❌ Kegagalan Fatal pada {name}: {e}")
                return []
                
        fetch_map = {
            "akademik_id": cls.fetch_academic_id,
            "akademik_en": cls.fetch_academic_en,
            "profesional_id": cls.fetch_professional_id,
            "profesional_en": cls.fetch_professional_en,
            "populer_id": cls.fetch_popular_id,
            "populer_en": cls.fetch_popular_en,
            "kreatif_id": cls.fetch_creative_id,
            "kreatif_en": cls.fetch_creative_en,
        }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_key = {}
            for key, func in fetch_map.items():
                checkpoint_path = os.path.join(CONFIG["CHECKPOINT_DIR"], f"checkpoint_{key}.json")
                loaded_from_checkpoint = False
                if os.path.exists(checkpoint_path):
                    try:
                        with open(checkpoint_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if len(data.get("data", [])) > 0: # Checkpoint exist and has data
                                texts = data["data"][:limit]
                                for t in texts:
                                    seen_hashes.add(hashlib.md5(t.encode('utf-8')).hexdigest())
                                corpus[key] = texts
                                logging.info(f"✅ Dimuat dari checkpoint: {key} ({len(texts)} docs).")
                                loaded_from_checkpoint = True
                    except Exception as e:
                        logging.warning(f"Gagal memuat checkpoint {key}: {e}")
                
                if not loaded_from_checkpoint:
                    future_to_key[executor.submit(func, limit)] = key
            
            for future in concurrent.futures.as_completed(future_to_key):
                style_key = future_to_key[future]
                corpus[style_key] = process_and_save(future, style_key, style_key)
            
        return corpus

def calculate_mattr(words, window_size=50):
    if not words: return 0.0
    if len(words) < window_size:
        return len(set(words)) / len(words)
    ttr_sum = 0
    num_windows = len(words) - window_size + 1
    for i in range(num_windows):
        window = words[i:i+window_size]
        ttr_sum += len(set(window)) / window_size
    return ttr_sum / num_windows

def detect_emotion(text, lang):
    text_lower = text.lower()
    if lang == "id":
        formal_markers = ["tersebut", "merupakan", "adalah", "berdasarkan", "signifikan", "penelitian"]
        informal_markers = ["banget", "sih", "dong", "gini", "gitu", "aja", "kok"]
        formal_count = sum(text_lower.count(m) for m in formal_markers)
        informal_count = sum(text_lower.count(m) for m in informal_markers)
        return "formal" if formal_count > informal_count else ("casual" if informal_count > formal_count else "neutral")
    else:
        formal_markers = ["furthermore", "therefore", "significant", "analysis", "demonstrates"]
        informal_markers = ["really", "awesome", "lol", "kinda", "gonna", "stuff"]
        formal_count = sum(text_lower.count(m) for m in formal_markers)
        informal_count = sum(text_lower.count(m) for m in informal_markers)
        return "formal" if formal_count > informal_count else ("casual" if informal_count > formal_count else "neutral")

EN_TRANSITIONS = ["however", "therefore", "furthermore", "in addition", "moreover", "subsequently", "meanwhile"]
EN_HEDGING = ["might", "could", "perhaps", "possibly", "suggests", "seems", "appears", "likely", "probably"]
ID_TRANSITIONS = ["namun", "oleh karena itu", "selain itu", "selanjutnya", "kemudian", "maka", "dengan demikian"]
ID_HEDGING = ["mungkin", "bisa jadi", "tampaknya", "kemungkinan", "sepertinya", "agaknya", "diperkirakan"]

# %% [markdown]
# ## 4. Modul Style Analyzer

# %%
class StyleAnalyzer:
    """Kelas untuk mengekstrak fitur linguistik menggunakan spaCy dan Stanza."""
    
    def __init__(self):
        logging.info("Memuat NLP Models (spaCy & Stanza) dengan akselerasi GPU...")
        
        # 1. Aktifkan GPU untuk spaCy
        spacy_gpu = spacy.prefer_gpu()
        if spacy_gpu:
            logging.info("✅ spaCy berjalan di atas GPU (T4)!")
        else:
            logging.info("⚠️ spaCy berjalan di atas CPU (GPU tidak terdeteksi).")
            
        self.nlp_en = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        
        # 2. Aktifkan GPU untuk Stanza (use_gpu=True)
        self.nlp_id = stanza.Pipeline("id", processors="tokenize,pos", verbose=False, use_gpu=True)
        logging.info("✅ Stanza dimuat dengan dukungan GPU (Hanya proses tokenize & pos).")

    def train_profile(self, texts, style_mode, lang):
        logging.info(f"Menganalisis {len(texts)} dokumen untuk gaya '{style_mode}_{lang}'...")
        features = []
        examples = []
        valid_texts = []
        
        for text in texts:
            if len(text.split()) < 50:
                continue
            if len(text) > 3000:
                text = text[:3000].rsplit(' ', 1)[0]
            
            valid_texts.append(text)
            if len(examples) < 2: examples.append(text)
                
        if not valid_texts:
            return None

        # 2. Batch Processing English (spaCy)
        if lang == "en":
            logging.info(f"-> Memproses {len(valid_texts)} teks English (spaCy Batch=256)...")
            textstat.set_lang("en")
            for doc in tqdm(self.nlp_en.pipe(valid_texts, batch_size=256), total=len(valid_texts)):
                sentences = list(doc.sents)
                words = [token.text for token in doc if not token.is_punct and not token.is_space]
                
                passive_sentences = sum(1 for sent in sentences if any(token.dep_ in ["auxpass", "nsubjpass"] for token in sent))
                active_ratio = 1.0 - (passive_sentences / max(len(sentences), 1))
                
                paragraphs = [p for p in doc.text.split('\n\n') if p.strip()]
                text_lower = doc.text.lower()
                trans_count = sum(text_lower.count(t) for t in EN_TRANSITIONS)
                hedge_count = sum(text_lower.count(h) for h in EN_HEDGING)
                
                if sentences and words:
                    features.append({
                        "avg_sentence_length": len(words) / len(sentences),
                        "flesch_reading_ease": textstat.flesch_reading_ease(doc.text),
                        "active_voice_ratio": active_ratio,
                        "emotion_register": detect_emotion(doc.text, "en"),
                        "lexical_diversity": calculate_mattr(words, 50),
                        "paragraph_length_avg": len(sentences) / max(len(paragraphs), 1),
                        "hedging_frequency": hedge_count / max(len(sentences), 1),
                        "transitions_count": trans_count
                    })
                    
        # 3. Batch Processing Indonesian (Stanza)
        elif lang == "id":
            batch_size = 64
            logging.info(f"-> Memproses {len(valid_texts)} teks Indonesian (Stanza Batch={batch_size})...")
            textstat.set_lang("id")
            for i in tqdm(range(0, len(valid_texts), batch_size)):
                chunk = valid_texts[i:i+batch_size]
                try:
                    docs = [self.nlp_id(d) for d in chunk]
                    for doc, orig_text in zip(docs, chunk):
                        sentences = doc.sentences
                        words = [word.text for sent in sentences for word in sent.words if word.upos != "PUNCT"]
                        
                        passive_verbs = sum(1 for sent in sentences for word in sent.words if word.text.lower().startswith("di") and word.upos == "VERB")
                        active_ratio = max(0.0, 1.0 - (passive_verbs / max(len(sentences), 1)))
                        
                        paragraphs = [p for p in orig_text.split('\n\n') if p.strip()]
                        text_lower = orig_text.lower()
                        trans_count = sum(text_lower.count(t) for t in ID_TRANSITIONS)
                        hedge_count = sum(text_lower.count(h) for h in ID_HEDGING)
                        
                        if sentences and words:
                            features.append({
                                "avg_sentence_length": len(words) / len(sentences),
                                "flesch_reading_ease": textstat.flesch_reading_ease(orig_text), 
                                "active_voice_ratio": active_ratio,
                                "emotion_register": detect_emotion(orig_text, "id"),
                                "lexical_diversity": calculate_mattr(words, 50),
                                "paragraph_length_avg": len(sentences) / max(len(paragraphs), 1),
                                "hedging_frequency": hedge_count / max(len(sentences), 1),
                                "transitions_count": trans_count
                            })
                    del docs
                    gc.collect()
                except Exception as e:
                    logging.error(f"Error Stanza batch {i}: {e}")
                
        def build_profile(feat_list, lang_id, ex):
            if not feat_list: return None
            df = pd.DataFrame(feat_list)
            std_val = float(df["avg_sentence_length"].std())
            if pd.isna(std_val): std_val = 0.0
            
            return {
                "user_id": f"mode_{style_mode}_{lang_id}",
                "style_mode": style_mode,
                "language": lang_id,
                "avg_sentence_length": float(df["avg_sentence_length"].mean()),
                "sentence_length_std": std_val,
                "flesch_reading_ease": float(df["flesch_reading_ease"].mean()),
                "active_voice_ratio": float(df["active_voice_ratio"].mean()),
                "emotion_register": df["emotion_register"].mode()[0] if not df["emotion_register"].empty else "neutral",
                "top_transitions": list(EN_TRANSITIONS) if lang_id == 'en' else list(ID_TRANSITIONS), # Simplified for profile
                "hedging_frequency": float(df["hedging_frequency"].mean()) if "hedging_frequency" in df else 0.0,
                "favorite_phrases": [],
                "lexical_diversity": float(df["lexical_diversity"].mean()) if "lexical_diversity" in df else 0.0,
                "paragraph_length_avg": float(df["paragraph_length_avg"].mean()) if "paragraph_length_avg" in df else 0.0,
                "few_shot_examples_id": ex if lang_id == "id" else [],
                "few_shot_examples_en": ex if lang_id == "en" else []
            }
            
        return build_profile(features, lang, examples)

# %% [markdown]
# ## 5. Pipeline Execution Manager

# %%
class PipelineManager:
    """Manajer yang mengontrol aliran dari pengumpulan data hingga penyimpanan JSON."""
    
    @staticmethod
    def run():
        os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
        
        # 1. Kumpulkan Data
        corpus = DataCollector.build_corpus(CONFIG["LIMIT_PER_STYLE"])
        
        # 2. Inisialisasi Analyzer
        analyzer = StyleAnalyzer()
        
        # 3. Analisis & Export
        for style_key, texts in corpus.items():
            if not texts:
                continue
            
            # Ekstrak mode dan lang dari style_key (misal: "akademik_id" -> mode="akademik", lang="id")
            parts = style_key.rsplit("_", 1)
            if len(parts) == 2:
                mode, lang = parts
            else:
                continue
                
            try:
                profile_data = analyzer.train_profile(texts, mode, lang)
                if profile_data:
                    file_path = os.path.join(CONFIG["OUTPUT_DIR"], f"{mode}_{lang}_style.json")
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(profile_data, f, indent=4)
                    logging.info(f"✅ Profil {mode} ({lang}) berhasil disimpan di: {file_path}")
            except Exception as e:
                logging.error(f"❌ Error saat melatih/menganalisis profil {style_key}: {e}")

        logging.info("✨ Seluruh pipeline selesai dieksekusi!")

# %% [markdown]
# ## 6. Run Pipeline
# Jalankan sel ini untuk memulai proses pelatihan secara keseluruhan.

# %%
if __name__ == "__main__":
    PipelineManager.run()

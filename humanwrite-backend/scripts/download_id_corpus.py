"""Download clean Indonesian text corpus for HumanWrite AI training.

Downloads diverse Indonesian articles from Wikipedia covering various topics
to build a natural, human-like style profile.
"""

import os
import sys
import re
import time
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))


def clean_text(text: str) -> str:
    """Remove web artifacts and clean up text."""
    # Remove common web artifacts
    patterns_to_remove = [
        r'Enter fullscreen mode',
        r'Exit fullscreen mode',
        r'Click here',
        r'Read more',
        r'Share this',
        r'Subscribe',
        r'\[edit\]',
        r'\[.*?\]',  # Remove bracketed references like [1], [2]
        r'─+',  # Remove horizontal lines
        r'•',  # Remove bullet points
        r'\*\*',  # Remove bold markers
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


# Indonesian Wikipedia articles covering diverse topics
WIKI_ARTICLES = [
    # Sejarah & Budaya
    "Sejarah_Indonesia",
    "Kerajaan_Majapahit",
    "Candi_Borobudur",
    "Proklamasi_Kemerdekaan_Indonesia",
    "Sumpah_Pemuda",
    "Perang_Diponegoro",
    "Batik",
    "Wayang",
    "Rendang",
    "Gamelan",
    
    # Geografi & Alam
    "Gunung_Merapi",
    "Danau_Toba",
    "Pulau_Komodo",
    "Hutan_hujan_tropis",
    "Gempa_bumi_dan_tsunami_Samudra_Hindia_2004",
    
    # Sains & Teknologi
    "Fotosintesis",
    "Pemanasan_global",
    "Energi_terbarukan",
    "Kecerdasan_buatan",
    "Internet",
    
    # Sosial & Ekonomi
    "Pendidikan_di_Indonesia",
    "Ekonomi_Indonesia",
    "Pertanian_di_Indonesia",
    "Urbanisasi",
    "Kemiskinan",
    
    # Kesehatan
    "Pandemi_COVID-19_di_Indonesia",
    "Demam_berdarah",
    "Vaksinasi",
    "Gizi",
    "Kesehatan_mental",
    
    # Tokoh
    "Soekarno",
    "Mohammad_Hatta",
    "Kartini",
    "Habibie",
    "Pramoedya_Ananta_Toer",
    
    # Olahraga & Hiburan
    "Bulu_tangkis",
    "Sepak_bola_Indonesia",
    "Film_Indonesia",
    "Musik_Indonesia",
    "Tari_tradisional_Indonesia",
    
    # Lain-lain  
    "Bahasa_Indonesia",
    "Pancasila",
    "Undang-Undang_Dasar_Negara_Republik_Indonesia_Tahun_1945",
    "Bencana_alam_di_Indonesia",
    "Transportasi_di_Indonesia",
    "Pariwisata_di_Indonesia",
    "Keanekaragaman_hayati_Indonesia",
    "Deforestasi_di_Indonesia",
    "Korupsi_di_Indonesia",
    "Hak_asasi_manusia_di_Indonesia",
]


def download_wikipedia_article(title: str) -> str:
    """Download a Wikipedia article in Indonesian."""
    import urllib.request
    import json
    
    url = f"https://id.wikipedia.org/api/rest_v1/page/summary/{title}"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HumanWriteAI/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            extract = data.get("extract", "")
            if len(extract) > 100:
                return extract
    except Exception:
        pass
    
    # Fallback: try full article extract
    url2 = f"https://id.wikipedia.org/w/api.php?action=query&titles={title}&prop=extracts&explaintext=1&format=json"
    try:
        req = urllib.request.Request(url2, headers={"User-Agent": "HumanWriteAI/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                extract = page_data.get("extract", "")
                if len(extract) > 100:
                    # Take first ~5000 chars to keep it manageable
                    return extract[:5000]
    except Exception as e:
        print(f"  Failed to download {title}: {e}")
    
    return ""


def main():
    corpus_dir = backend_dir / "data" / "training_corpus_id"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("HumanWrite AI — Indonesian Corpus Downloader")
    print("=" * 60)
    print(f"\nTarget directory: {corpus_dir}")
    print(f"Articles to download: {len(WIKI_ARTICLES)}\n")
    
    success = 0
    for i, title in enumerate(WIKI_ARTICLES, 1):
        print(f"[{i}/{len(WIKI_ARTICLES)}] Downloading: {title}...", end=" ")
        text = download_wikipedia_article(title)
        
        if text:
            cleaned = clean_text(text)
            if len(cleaned) > 200:
                filename = f"id_{title.lower().replace(' ', '_')[:50]}.txt"
                filepath = corpus_dir / filename
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                success += 1
                print(f"OK ({len(cleaned)} chars)")
            else:
                print("SKIPPED (too short)")
        else:
            print("FAILED")
        
        time.sleep(0.5)  # Be polite to Wikipedia
    
    print(f"\n{'=' * 60}")
    print(f"Downloaded {success}/{len(WIKI_ARTICLES)} articles")
    print(f"Corpus directory: {corpus_dir}")
    print(f"\nNext step: run train_global_style_id.py to train the profile")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

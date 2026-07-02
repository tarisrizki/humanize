import asyncio
import re
import statistics
from typing import List

from dotenv import load_dotenv
load_dotenv()
import logging
logging.basicConfig(level=logging.DEBUG)

from app.core.pipeline import apply_style
from app.core.text_utils import _enforce_min_sentences
from app.models.style_profile import StyleProfile

# 3 paragraf teks akademik
ACADEMIC_DRAFT = """
Kecerdasan buatan (AI) telah berkembang secara signifikan dalam dekade terakhir dan telah diterapkan dalam berbagai bidang industri. Perkembangan ini tidak hanya didorong oleh peningkatan kapasitas komputasi, tetapi juga ketersediaan data yang besar. Berbagai penelitian telah menunjukkan bahwa algoritma pembelajaran mesin mampu melampaui kemampuan manusia dalam tugas-tugas spesifik seperti pengenalan pola dan analisis prediktif.

Namun demikian, implementasi AI di sektor publik masih menghadapi berbagai tantangan yang kompleks. Salah satu tantangan utama adalah masalah privasi dan keamanan data pengguna. Selain itu, ada kekhawatiran terkait bias algoritmik yang dapat menghasilkan keputusan diskriminatif terhadap kelompok marginal. Oleh karena itu, diperlukan regulasi yang komprehensif untuk memitigasi risiko tersebut.

Ke depan, kolaborasi antara pemerintah, akademisi, dan praktisi industri akan menjadi sangat penting. Hal ini diperlukan untuk membangun kerangka etika yang kuat dalam pengembangan teknologi cerdas. Secara keseluruhan, pemanfaatan AI yang bertanggung jawab dapat memberikan manfaat optimal bagi masyarakat tanpa mengorbankan nilai-nilai kemanusiaan.
"""

AI_BLACKLIST = [
    "merupakan", "memiliki", "berbagai", "sehingga", "serta",
    "tersebut", "selain itu", "oleh karena itu", "dengan demikian",
    "hal ini", "dapat disimpulkan", "secara keseluruhan",
    "sangat penting", "perlu dicatat", "di sisi lain", "adapun",
    "tentunya", "pastinya", "dalam hal ini"
]

def count_sentences(text: str) -> List[int]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    counts = []
    for p in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', p.strip())
        sentences = [s for s in sentences if s.strip()]
        counts.append(len(sentences))
    return counts

def check_blacklist(text: str) -> List[str]:
    found = []
    text_lower = text.lower()
    for word in AI_BLACKLIST:
        if re.search(r'\b' + word + r'\b', text_lower):
            found.append(word)
    return found

async def main():
    print("=== PENGUJIAN 3 PARAGRAF TEKS AKADEMIK ===")
    import os
    try:
        from app.storage.json_store import load_json
        profile_path = os.path.join(os.path.dirname(__file__), "data", "profiles", "akademik_id_style.json")
        profile_data = load_json(profile_path)
        style = StyleProfile(**profile_data)
        print("Loaded akademik_id_style.json successfully!")
    except Exception as e:
        print(f"Failed to load profile: {e}")
        style = StyleProfile(user_id="test", language="id", style_mode="akademik")
    
    for i in range(1, 6):
        print(f"\n[ RUN {i} ]")
        try:
            result = await apply_style(ACADEMIC_DRAFT, style)
            output = result.final_text
            
            sents_count = count_sentences(output)
            blacklist_found = check_blacklist(output)
            
            print(f"Kalimat per paragraf: {sents_count}")
            if all(c >= 4 for c in sents_count):
                print("OK PASS: Semua paragraf >= 4 kalimat")
            else:
                print("FAIL: Ada paragraf < 4 kalimat")
                
            if not blacklist_found:
                print("OK PASS: Tidak ada kata blacklist")
            else:
                print(f"FAIL: Ditemukan kata blacklist: {blacklist_found}")
                
            print(f"Output Snippet (Awal): {output[:150]}...")
        except Exception as e:
            print(f"Error pada run {i}: {e}")

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import os
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.config import settings
from app.core.writing_engine import apply_style
from app.models.style_profile import StyleProfile
from app.storage.json_store import load_json
from app.core.detection_scorer import score_humanness

async def main():
    # Load the global style profile we just created
    profile_path = backend_dir / "data" / "profiles" / "global_style.json"
    profile_data = load_json(profile_path)
    style = StyleProfile(**profile_data)
    
    draft = "Pada akhir November hingga awal Desember 2025, Provinsi Aceh mengalami banjir besar akibat curah hujan yang sangat tinggi dan berlangsung selama beberapa hari berturut-turut. Hujan lebat menyebabkan sejumlah sungai meluap sehingga merendam permukiman, lahan pertanian, fasilitas umum, serta ruas jalan di berbagai daerah. Banjir melanda sekitar 18 kabupaten/kota, dengan wilayah seperti Aceh Utara, Aceh Timur, Aceh Tamiang, Pidie, Bireuen, dan Aceh Selatan menjadi beberapa daerah yang terdampak cukup parah. Penyebab utama banjir adalah kombinasi hujan ekstrem, meluapnya sungai, kapasitas drainase yang terbatas, serta berkurangnya daerah resapan air."
    
    print("\n--- Rewriting using Groq (llama-3.3-70b-versatile) ---")
    result = await apply_style(draft, style)
    print("\n--- Rewritten Result ---")
    print(result.final_text)
    
    score = score_humanness(result.final_text, lang="id")
    print(f"\n--- Burstiness Score ---")
    print(score)

if __name__ == "__main__":
    asyncio.run(main())

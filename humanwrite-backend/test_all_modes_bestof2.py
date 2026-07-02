import asyncio
import os
import sys

# Tambahkan backend dir ke sys.path agar import absolute app.* berfungsi
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()


from app.models.style_profile import StyleProfile

# Draft akademik untuk diujikan
DRAFT = """
Dalam tahun-tahun terakhir, kecerdasan buatan telah berkembang dengan sangat pesat. Hal ini tidak terlepas dari peran algoritma deep learning dan kemajuan perangkat keras komputer. Oleh karena itu, kita dapat melihat implementasi AI di berbagai macam bidang industri mulai dari kesehatan hingga manufaktur.

Secara keseluruhan, pemanfaatan AI memiliki dampak positif dalam meningkatkan efisiensi operasional. Sebagai contoh, di bidang kesehatan, AI digunakan untuk mendiagnosis penyakit dengan lebih cepat. Di sisi lain, industri manufaktur menggunakan AI untuk otomatisasi proses produksi sehingga meminimalisir kesalahan manusia.

Dapat disimpulkan bahwa kecerdasan buatan merupakan teknologi yang sangat penting untuk masa depan. Dengan demikian, perlu dicatat bahwa pengembangannya harus diiringi dengan regulasi yang tepat. Hal tersebut bertujuan untuk memastikan bahwa AI memberikan manfaat maksimal tanpa menimbulkan kerugian yang tidak diinginkan.
"""

async def run_tests():
    print("=== PENGUJIAN BEST-OF-2 UNTUK SEMUA MODE (3x) ===")
    modes = ["akademik", "profesional", "kreatif", "populer"]
    
    for mode in modes:
        print(f"\n--- MODE: {mode.upper()} ---")
        try:
            from app.storage.json_store import load_json
            profile_path = os.path.join(os.path.dirname(__file__), "data", "profiles", f"{mode}_id_style.json")
            profile_data = load_json(profile_path)
            style = StyleProfile(**profile_data)
        except Exception as e:
            print(f"Failed to load profile for {mode}: {e}")
            style = StyleProfile(user_id="test", language="id", style_mode=mode)
        
        for i in range(1, 4):
            try:
                from app.core.pipeline import apply_style
                from app.core.text_utils import _score_human_likelihood, _clean_input_draft
                
                result = await apply_style(DRAFT, style)
                text = result.final_text
                
                clean_draft = _clean_input_draft(DRAFT)
                final_score = _score_human_likelihood(text, clean_draft, "id")
                
                sents_count = []
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                for p in paragraphs:
                    import re
                    sents = [s for s in re.split(r'(?<=[.!?])\s+', p.strip()) if s.strip()]
                    sents_count.append(len(sents))
                
                print(f"[{mode.upper()} Run {i}] Skor Human Likelihood: {final_score}")
                print(f"Kalimat per paragraf: {sents_count}")
                if final_score < 0.55:
                    print(f"WARNING: Skor masih < 0.55!")
                
            except Exception as e:
                print(f"[{mode.upper()} Run {i}] Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_tests())

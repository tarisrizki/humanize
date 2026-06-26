import json
from pathlib import Path

DATA_DIR = Path("humanwrite-backend/data/profiles")

def update_akademik():
    path = DATA_DIR / "akademik_style.json"
    if not path.exists(): return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["avg_sentence_length"] = 24.5
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Fixed akademik_style.json")

def update_profesional():
    path = DATA_DIR / "profesional_style.json"
    if not path.exists(): return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["few_shot_examples_id"] = [
        "Laba bersih PT Bank Central Asia Tbk (BCA) dan entitas anak tumbuh 19,4% secara tahunan (YoY) menjadi Rp48,6 triliun sepanjang 2023. Kinerja positif ini didorong oleh pertumbuhan kredit yang solid, perbaikan kualitas pinjaman, serta peningkatan efisiensi operasional sejalan dengan membaiknya kondisi ekonomi nasional pascapandemi.",
        "Pertumbuhan kredit perseroan didukung oleh permintaan yang kuat dari berbagai segmen. Kredit korporasi mencatat pertumbuhan tertinggi, meningkat 15,0% YoY mencapai Rp327,3 triliun. Sejalan dengan hal tersebut, kredit komersial dan UKM juga menunjukkan tren positif dengan kenaikan 7,5% YoY menjadi Rp222,8 triliun, didorong oleh akselerasi aktivitas bisnis di sektor riil.",
        "Dalam rangka mendukung transisi menuju ekonomi hijau, portofolio pembiayaan berkelanjutan perseroan tumbuh 10,6% YoY mencapai Rp202,6 triliun, atau berkontribusi sebesar 24,8% terhadap total portofolio pembiayaan. Capaian ini menunjukkan komitmen strategis perseroan dalam menerapkan prinsip environmental, social, and governance (ESG) dalam operasional bisnisnya.",
        "Meskipun menghadapi tantangan volatilitas pasar global dan tekanan inflasi, likuiditas bank tetap terjaga pada tingkat yang sangat memadai. Rasio cakupan likuiditas (LCR) berada di angka 205,8%, jauh di atas batas minimum regulasi yang ditetapkan oleh Otoritas Jasa Keuangan (OJK). Hal ini memberikan ruang gerak yang luas bagi bank untuk mengoptimalkan penyaluran kredit di tahun berjalan."
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Fixed profesional_style.json")

def update_kreatif():
    path = DATA_DIR / "kreatif_style.json"
    if not path.exists(): return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["few_shot_examples_id"] = [
        "Hujan sore itu turun tidak terlalu deras, hanya gerimis yang cukup untuk membuat aroma tanah basah menyeruak di udara. Di balik jendela kaca kedai kopi yang berembun, Risa menatap nanar ke arah persimpangan jalan. Lampu-lampu kendaraan yang lalu-lalang tampak seperti kunang-kunang raksasa yang kebingungan mencari arah pulang. Jemarinya sibuk mengetuk-ngetuk meja kayu jati di hadapannya, seirama dengan detak jarum jam yang seolah enggan beranjak.",
        "Kopi di cangkirnya sudah lama mendingin, kehilangan kepulan asap yang memikat. Sesekali ia merapatkan mantel wolnya, mencoba mengusir hawa dingin yang perlahan menyusup lewat celah pintu yang terbuka. Namun, dingin sesungguhnya bukan berasal dari angin sore, melainkan dari kenyataan bahwa seseorang yang ia tunggu sejak dua jam lalu, mungkin tak akan pernah datang.",
        "Dari kejauhan, sayup-sayup terdengar suara pengamen jalanan menyanyikan lagu lama dengan gitar kopong yang nadanya sedikit fals. Nada-nada sumbang itu justru terasa sangat pas dengan suasana hatinya saat ini—berantakan dan kehilangan ritme. Ia menarik napas panjang, membiarkan udara lembap memenuhi paru-parunya, sebelum akhirnya memutuskan untuk bangkit berdiri.",
        "Sambil melangkah keluar dari kedai, ia membiarkan rintik hujan membasahi wajahnya. Jalanan aspal yang memantulkan cahaya neon minimarket di seberang seolah menjadi saksi bisu langkah kakinya yang berat. Ia tahu, mulai malam ini, ia harus belajar berjalan menembus badai sendirian, tanpa ada lagi tangan yang bisa ia genggam saat guntur menyambar."
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Fixed kreatif_style.json")

if __name__ == "__main__":
    update_akademik()
    update_profesional()
    update_kreatif()

import asyncio
import os
from app.config import settings
from app.core.writing_engine import apply_style
from app.models.style_profile import StyleProfile

async def main():
    # Setup style
    style = StyleProfile(
        id="test",
        name="Test",
        language="id",
        emotion_register="Casual",
        formality_level=3,
        favorite_phrases=[],
        sentence_length_variance=5.0
    )
    
    draft = """Pada akhir November hingga awal Desember 2025, Provinsi Aceh dilanda banjir besar yang mengakibatkan kerusakan luas. Namun, hujan lebat yang turun selama beberapa hari berturut-turut menyebabkan sejumlah sungai meluap, sehingga permukiman, lahan pertanian, fasilitas umum, serta ruas jalan di berbagai daerah terendam. Banjir ini melanda sekitar 18 kabupaten/kota, dengan wilayah seperti Aceh Utara, Aceh Timur, Aceh Tamiang, Pidie, Bireuen, dan Aceh Selatan menjadi beberapa daerah yang terdampak cukup parah. Penyebab utama banjir ini adalah kombinasi hujan ekstrem, meluapnya sungai, kapasitas drainase yang terbatas, serta berkurangnya daerah resapan air.

Dampak banjir sangat luas terhadap kehidupan masyarakat, dengan ribuan rumah terendam, ratusan ribu warga terpaksa mengungsi, dan berbagai fasilitas umum seperti sekolah, puskesmas, jembatan, serta jalan mengalami kerusakan. Sektor pertanian juga mengalami kerugian besar karena sawah, kebun, dan tambak terendam banjir, sehingga mengganggu produksi pangan dan mata pencaharian masyarakat. Dan singkat cerita, kerugian akibat berbagai bencana di Aceh sepanjang tahun 2025 diperkirakan mencapai ratusan miliar rupiah, dengan banjir sebagai penyumbang dampak terbesar.

Dalam penanganannya, pemerintah melalui BPBA, BNPB, TNI, Polri, Basarnas, serta pemerintah daerah melakukan evakuasi warga, mendirikan posko pengungsian, menyediakan dapur umum, menyalurkan bantuan logistik dan layanan kesehatan, serta memperbaiki infrastruktur yang rusak setelah banjir surut. Peristiwa ini menjadi pengingat akan pentingnya peningkatan sistem drainase, normalisasi sungai, pelestarian daerah resapan air, serta penguatan sistem peringatan dini dan kesiapsiagaan masyarakat untuk mengurangi risiko banjir serupa di masa mendatang."""

    print("Running engine...")
    res = await apply_style(draft, style)
    print("---------------------------------")
    print(res.final_text)
    print("---------------------------------")
    print("Changes:", res.changes_made)

if __name__ == "__main__":
    asyncio.run(main())

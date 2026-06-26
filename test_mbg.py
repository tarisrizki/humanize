import asyncio
import os
import sys

# Adjust path so we can import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'humanwrite-backend')))

from app.core.writing_engine import apply_style
from app.models.style_profile import StyleProfile

MBG_TEXT = """
Program Makan Bergizi Gratis (MBG) merupakan salah satu program pemerintah yang bertujuan meningkatkan kualitas gizi masyarakat, khususnya peserta didik, guna mendukung pertumbuhan, kesehatan, dan kemampuan belajar. Program ini diharapkan dapat membantu mengurangi angka kekurangan gizi, meningkatkan konsentrasi siswa di sekolah, serta mendukung upaya pembangunan sumber daya manusia yang berkualitas. Meskipun memiliki tujuan yang positif, pelaksanaannya masih menghadapi berbagai tantangan di lapangan.

Salah satu permasalahan utama dalam pelaksanaan MBG adalah distribusi makanan yang belum merata. Beberapa sekolah, terutama yang berada di daerah terpencil, mengalami keterlambatan pengiriman atau keterbatasan pasokan bahan makanan. Kondisi tersebut menyebabkan pelaksanaan program tidak berjalan secara konsisten dan mengurangi efektivitas manfaat yang diharapkan bagi para penerima.

Selain itu, kualitas dan keamanan pangan juga menjadi perhatian penting. Dalam beberapa kasus, ditemukan makanan yang kurang memenuhi standar kebersihan atau tidak lagi layak dikonsumsi akibat proses penyimpanan dan distribusi yang kurang baik. Kondisi ini dapat menimbulkan keluhan dari peserta didik dan orang tua serta berpotensi mengganggu kesehatan apabila pengawasan tidak dilakukan secara ketat.

Permasalahan lainnya berkaitan dengan pengelolaan anggaran dan koordinasi antarinstansi. Pelaksanaan MBG melibatkan banyak pihak, mulai dari pemerintah pusat, pemerintah daerah, sekolah, hingga penyedia makanan. Apabila koordinasi dan pengawasan tidak berjalan dengan baik, dapat terjadi kendala dalam proses pengadaan, penyaluran, maupun pelaporan penggunaan anggaran sehingga tujuan program menjadi kurang optimal.

Untuk mengatasi berbagai permasalahan tersebut, diperlukan evaluasi dan perbaikan secara berkelanjutan. Pemerintah perlu memperkuat sistem pengawasan terhadap kualitas makanan, meningkatkan efektivitas distribusi, memperbaiki koordinasi antarinstansi, serta memastikan pengelolaan anggaran dilakukan secara transparan dan akuntabel. Dengan langkah-langkah tersebut, Program Makan Bergizi Gratis diharapkan dapat memberikan manfaat yang lebih optimal dalam meningkatkan status gizi, kesehatan, dan kualitas pendidikan peserta didik di Indonesia.
"""

async def main():
    profile = StyleProfile(
        user_id="test",
        style_mode="populer",
        language="id",
        few_shot_examples_id=[],
        few_shot_examples_en=[],
        avg_sentence_length=15.0,
        burstiness_score=0.5,
        word_reduction_rate=0.1
    )
    
    print("Mulai proses MBG text...")
    result = await apply_style(MBG_TEXT, profile)
    print("\n--- HASIL HUMANIZE ---\n")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

Buatkan aplikasi HumanWrite AI yang terdiri dari:

1. Backend REST API menggunakan FastAPI (Python 3.11+, uv)
2. GUI minimalis menggunakan Streamlit untuk testing dan demo

Seluruh kode backend disimpan di folder `humanwrite-backend/`. GUI adalah satu file `app_gui.py` di root proyek.

### DESKRIPSI PRODUK
HumanWrite AI memproses draft dari LLM agar sesuai gaya pribadi pengguna. Pengguna mengunggah 1–3 dokumen asli, sistem menganalisis gaya penulisan (Style Profile), lalu saat pengguna memberikan draft teks, sistem menulis ulang draft tersebut agar memiliki gaya yang sama dengan dokumen asli pengguna. Output adalah teks final, skor kecocokan gaya, dan daftar perubahan.

### ARSITEKTUR SISTEM
Backend dijalankan dengan `uvicorn` di port 8000, GUI Streamlit di port 8501. GUI melakukan HTTP request ke backend.

---

## BAGIAN 1: BACKEND (sama seperti sebelumnya, diringkas)

### Struktur Folder Backend
humanwrite-backend/
├── app/
│ ├── main.py
│ ├── config.py
│ ├── api/v1/endpoints/
│ │ ├── upload.py
│ │ ├── analyze.py
│ │ └── process.py
│ ├── core/
│ │ ├── style_analyzer.py
│ │ └── writing_engine.py
│ ├── models/
│ │ ├── style_profile.py
│ │ └── schemas.py
│ └── storage/
│ └── json_store.py
├── data/ (subfolder corpus, profiles, outputs)
├── app_gui.py
├── .env.example
├── pyproject.toml
└── README.md

text

### Rincian Komponen Backend (ringkas)
- `config.py`: Settings dari .env (GEMINI_API_KEY, SPACY_MODEL="en_core_web_sm").
- `json_store.py`: Fungsi `save_json(path, data)`, `load_json(path)`.
- `style_profile.py`: Pydantic model `StyleProfile` dengan field: user_id, avg_sentence_length, sentence_length_std, flesch_reading_ease, active_voice_ratio, top_transitions, hedging_frequency, emotion_register, favorite_phrases, lexical_diversity, paragraph_length_avg.
- `style_analyzer.py`: `analyze_style(texts: list[str]) -> StyleProfile` menggunakan spaCy dan textstat.
- `writing_engine.py`: `apply_style(draft, style) -> dict` menggunakan PydanticAI dengan Gemini 2.5 Flash. System prompt berisi seluruh parameter StyleProfile, instruksi menulis ulang tanpa mengubah makna. Output Pydantic model `ProcessedText` (final_text, voice_match_score, changes_made).
- `upload.py`: POST /upload terima user_id dan multiple files (txt, docx, pdf), ekstrak teks, simpan corpus ke disk.
- `analyze.py`: POST /analyze terima user_id, proses semua teks di corpus, simpan StyleProfile ke `/data/profiles/{user_id}.json`, kembalikan profile.
- `process.py`: POST /process terima user_id dan draft, load profile, panggil writing_engine, kembalikan ProcessedText.
- `main.py`: FastAPI app dengan CORS allow all, lifespan load spaCy model, include routers.

### Requirements backend
- fastapi, uvicorn[standard], pydantic-ai, spacy, textstat, python-docx, pypdf, pydantic-settings, python-dotenv, httpx (untuk test)
- spacy model `en_core_web_sm` harus diunduh otomatis saat startup atau lewat `python -m spacy download en_core_web_sm`.
- Gemini API key diambil dari env GEMINI_API_KEY.

---

## BAGIAN 2: GUI MINIMALIS DENGAN STREAMLIT

File: `humanwrite-backend/app_gui.py`

### Tujuan
GUI sederhana untuk melakukan alur: upload, analyze, dan humanize draft. Tampilan minimalis dengan Streamlit.

### Komponen GUI
1. **Judul**: "HumanWrite AI – Voice Preservation Demo"
2. **Sidebar**: Input `user_id` (default "demo"), tombol "Reset Data" (menghapus folder data/user_id).
3. **Tab 1: Upload Dokumen Asli**
   - File uploader mendukung multiple files (txt, docx, pdf).
   - Tombol "Upload & Analyze" yang memanggil POST /upload lalu otomatis POST /analyze.
   - Menampilkan StyleProfile yang dihasilkan sebagai JSON expander atau metrik singkat.
4. **Tab 2: Humanize Draft**
   - Text area untuk memasukkan draft.
   - Tombol "Humanize Draft" memanggil POST /process.
   - Menampilkan output dalam dua kolom: Draft Asli vs Teks Final (bisa menggunakan `st.columns`).
   - Menampilkan Voice Match Score dan daftar perubahan (changes_made) di bawahnya.
   - Copy button untuk teks final (bisa pakai `st.code` atau `st.write` dengan clipboard library, minimal gunakan `st.text_area` readonly).

### Logika Penting
- Base URL backend diambil dari environment variable `BACKEND_URL`, default `http://localhost:8000`.
- Gunakan `requests` untuk HTTP call.
- Tangani error jika backend tidak berjalan atau profile belum ada.
- Setelah upload & analyze, simpan state `profile_ready = True` menggunakan `st.session_state` agar tab Humanize bisa diaktifkan.

### Instalasi & Menjalankan
- `uv add streamlit requests`
- Jalankan backend: `uv run uvicorn app.main:app --reload --port 8000`
- Jalankan GUI: `uv run streamlit run app_gui.py --server.port 8501`

---

## PERINTAH UNTUK AI BUILDER
1. Buat struktur folder di atas.
2. Tulis seluruh kode backend sesuai spesifikasi, pastikan semua endpoint berfungsi.
3. Buat `pyproject.toml` dengan dependency yang disebutkan.
4. Buat `.env.example` dengan `GEMINI_API_KEY=your_api_key_here` dan `SPACY_MODEL=en_core_web_sm`.
5. Buat `app_gui.py` dengan Streamlit mengikuti spesifikasi GUI.
6. Pastikan kode siap dijalankan dengan instruksi jelas di README.

Jangan lupa: backend menggunakan PydanticAI dengan Gemini untuk writing_engine. Jika ada kode yang memerlukan inisialisasi agen, gunakan pola:

```python
from pydantic_ai import Agent
agent = Agent(
    model="gemini-2.5-flash",
    system_prompt=system_prompt,
    result_type=ProcessedText
)
result = await agent.run(draft)
Gunakan API key dari config.

Sekarang, buat seluruh proyek.

text

---

Superprompt ini akan menghasilkan **satu proyek utuh** yang bisa langsung kamu jalankan di laptop: backend di satu terminal, GUI Streamlit di terminal lain. Kamu bisa langsung mengunggah dokumen, menganalisis gaya, dan mencoba humanisasi draft.

Apakah ada yang ingin kamu sesuaikan lagi sebelum dipakai?
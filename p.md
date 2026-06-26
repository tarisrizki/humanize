Implementasikan fitur Output A vs B (Standard vs Judge Enhanced)
dalam satu commit. Ini melibatkan backend, API, DB, dan UI.

═══════════════════════════════════════════════════════
ARSITEKTUR YANG DIPILIH: Conditional Enhancement (Opsi B)
- Standard (A) muncul duluan via streaming yang sudah ada
- Enhanced (B) dihitung otomatis setelah A selesai
- Pass 3 hanya jalan jika judge_score < 65
- Jika score ≥ 65: Enhanced = Standard (sudah optimal)
═══════════════════════════════════════════════════════


═══════════════════════════════════════════════════════
PERUBAHAN 1: Tambah endpoint POST /process/enhance
File: humanwrite-backend/app/api/v1/endpoints/process.py
═══════════════════════════════════════════════════════

Tambahkan Pydantic model baru:

class EnhanceRequest(BaseModel):
    original_text: str
    standard_output: str
    style_mode: str = "populer"
    language:   str = "id"


class EnhancedResult(BaseModel):
    enhanced_text:     str
    judge_score_before: float   # skor Standard
    judge_score_after:  float   # skor Enhanced
    worst_sentence:    str
    improved:          bool     # True jika enhanced != standard
    trigram_overlap:   float


Tambahkan endpoint baru (SETELAH endpoint /process yang ada):

@router.post("/process/enhance", response_model=EnhancedResult)
async def enhance_output(request: EnhanceRequest):
    """
    Tahap 2: Terima standard output, jalankan LLM Judge,
    lalu Pass 3 jika judge_score < 65.
    Dipanggil setelah /process selesai streaming.
    """
    from app.core.evaluator import run_llm_judge
    from app.core.writing_engine import (
        _apply_post_processing,
        _programmatic_sentence_humanize,
        _validate_paragraph_count,
        _generate_changes_made,
        check_trigram_overlap,
        _clean_input_draft,
        _count_paragraphs,
        _build_system_prompt,
    )
    from app.storage.json_store import load_style_profile
    from pydantic_ai import Agent
    from pydantic_ai.models.fallback import FallbackModel

    clean_original = _clean_input_draft(request.original_text)
    paragraph_count = _count_paragraphs(clean_original)

    # Step 1: Judge Standard output
    judge_result = await run_llm_judge(
        original_text=clean_original,
        humanized_text=request.standard_output,
        style_mode=request.style_mode,
        language=request.language,
    )

    score_before   = judge_result.get("overall_score", 100)
    worst_sentence = judge_result.get("worst_sentence", "")

    # Step 2: Jika score < 65 → Pass 3 targeted fix
    if score_before < 65 and worst_sentence:
        profile_data = load_style_profile(request.style_mode)
        profile_data["style_mode"] = request.style_mode
        from app.models.style_profile import StyleProfile
        style = StyleProfile(**profile_data)
        system_prompt = _build_system_prompt(style, paragraph_count)

        agent = Agent(
            model=FallbackModel(
                "groq:llama-3.3-70b-versatile",
                "groq:llama-3.1-8b-instant",
            ),
            system_prompt=system_prompt,
        )

        pass3_msg = (
            f"Teks berikut mendapat skor {score_before}/100 "
            f"dari evaluator karena kalimat ini masih terasa AI:\n"
            f"\"{worst_sentence}\"\n\n"
            f"Tulis ulang SELURUH teks — fokus perbaiki kalimat "
            f"di atas agar lebih natural. Pertahankan semua fakta "
            f"dan jumlah paragraf ({paragraph_count} paragraf).\n\n"
            f"{request.standard_output}"
        )

        try:
            result3 = await agent.run(
                pass3_msg,
                model_settings={"temperature": 1.0},
            )
            enhanced_raw = str(result3.output).strip()
        except Exception:
            enhanced_raw = request.standard_output

        # Post-processing Pass 3
        enhanced = _apply_post_processing(
            enhanced_raw, request.language, request.style_mode
        )
        enhanced = _programmatic_sentence_humanize(
            enhanced, request.language, request.style_mode
        )
        enhanced = _validate_paragraph_count(
            enhanced, paragraph_count, clean_original
        )

        # Judge Enhanced
        judge_after = await run_llm_judge(
            original_text=clean_original,
            humanized_text=enhanced,
            style_mode=request.style_mode,
            language=request.language,
        )
        score_after = judge_after.get("overall_score", score_before)
        improved    = score_after > score_before

        # Gunakan enhanced hanya jika benar-benar lebih baik
        final_enhanced = enhanced if improved else request.standard_output
        final_score    = score_after if improved else score_before

    else:
        # Score sudah bagus — Enhanced = Standard
        final_enhanced = request.standard_output
        final_score    = score_before
        improved       = False

    trigram = check_trigram_overlap(clean_original, final_enhanced)

    return EnhancedResult(
        enhanced_text=final_enhanced,
        judge_score_before=score_before,
        judge_score_after=final_score,
        worst_sentence=worst_sentence,
        improved=improved,
        trigram_overlap=trigram,
    )


═══════════════════════════════════════════════════════
PERUBAHAN 2: Update SQLite schema di evaluator.py
Tambah kolom untuk Enhanced output dan GPTZero comparison
═══════════════════════════════════════════════════════

Di CREATE TABLE IF NOT EXISTS evaluations,
tambahkan kolom baru setelah kolom yang sudah ada:

    -- Enhanced output (Output B)
    enhanced_text           TEXT,
    enhanced_judge_score    REAL,
    enhanced_trigram        REAL,
    judge_score_standard    REAL,

    -- GPTZero comparison (A vs B) — diisi manual user
    gptzero_std_ai          INTEGER,
    gptzero_std_mixed       INTEGER,
    gptzero_std_human       INTEGER,
    gptzero_enh_ai          INTEGER,
    gptzero_enh_mixed       INTEGER,
    gptzero_enh_human       INTEGER,

Tambahkan juga field-field ini ke class EvaluationRecord:

    enhanced_text:        Optional[str]   = None
    enhanced_judge_score: Optional[float] = None
    enhanced_trigram:     Optional[float] = None
    judge_score_standard: Optional[float] = None
    gptzero_std_ai:       Optional[int]   = None
    gptzero_std_mixed:    Optional[int]   = None
    gptzero_std_human:    Optional[int]   = None
    gptzero_enh_ai:       Optional[int]   = None
    gptzero_enh_mixed:    Optional[int]   = None
    gptzero_enh_human:    Optional[int]   = None

Tambahkan ALTER TABLE migration untuk DB yang sudah exist:

    for col in [
        "enhanced_text TEXT",
        "enhanced_judge_score REAL",
        "enhanced_trigram REAL",
        "judge_score_standard REAL",
        "gptzero_std_ai INTEGER",
        "gptzero_std_mixed INTEGER",
        "gptzero_std_human INTEGER",
        "gptzero_enh_ai INTEGER",
        "gptzero_enh_mixed INTEGER",
        "gptzero_enh_human INTEGER",
    ]:
        try:
            conn.execute(
                f"ALTER TABLE evaluations ADD COLUMN {col}"
            )
        except Exception:
            pass  # kolom sudah ada


═══════════════════════════════════════════════════════
PERUBAHAN 3: Update endpoint PATCH /gptzero di evaluate.py
Tambah field GPTZero Standard vs Enhanced
═══════════════════════════════════════════════════════

Update Pydantic model GPTZeroUpdate:

class GPTZeroUpdate(BaseModel):
    record_id: int

    # Standard (Output A) — sudah ada, pertahankan
    gptzero_ai:    Optional[int] = None
    gptzero_mixed: Optional[int] = None
    gptzero_human: Optional[int] = None

    # Enhanced (Output B) — baru
    gptzero_std_ai:    Optional[int] = None
    gptzero_std_mixed: Optional[int] = None
    gptzero_std_human: Optional[int] = None
    gptzero_enh_ai:    Optional[int] = None
    gptzero_enh_mixed: Optional[int] = None
    gptzero_enh_human: Optional[int] = None

Update endpoint PATCH agar simpan semua field baru ke DB.

Tambahkan juga endpoint GET /compare/{record_id}
untuk ambil data perbandingan A vs B:

@router.get("/compare/{record_id}")
def get_comparison(record_id: int):
    """Ambil data A vs B untuk ditampilkan di History."""
    # Query DB dan return semua field standard + enhanced
    ...


═══════════════════════════════════════════════════════
PERUBAHAN 4: Update app_gui.py — Tab Humanize
Tampilkan Output A dan B side-by-side
═══════════════════════════════════════════════════════

Di tab Humanize (tab1), setelah streaming Output A selesai,
otomatis panggil POST /process/enhance dan tampilkan hasilnya.

Struktur UI yang diinginkan:

# Setelah streaming selesai, simpan ke session state
st.session_state["standard_text"]   = output_a
st.session_state["standard_trigram"] = trigram_a  # dari changes_made

# Panggil /enhance di background
with st.spinner("✨ Menjalankan Judge Enhancement..."):
    enhance_resp = requests.post(
        f"{API_URL}/api/v1/process/enhance",
        json={
            "original_text":   st.session_state["current_draft"],
            "standard_output": output_a,
            "style_mode":      selected_mode,
            "language":        detected_lang,
        },
        timeout=60,
    )
    enh_data = enhance_resp.json()

st.session_state["enhanced_text"]        = enh_data["enhanced_text"]
st.session_state["judge_score_before"]   = enh_data["judge_score_before"]
st.session_state["judge_score_after"]    = enh_data["judge_score_after"]
st.session_state["enhanced_trigram"]     = enh_data["trigram_overlap"]
st.session_state["enhanced_improved"]    = enh_data["improved"]
st.session_state["worst_sentence"]       = enh_data["worst_sentence"]

# Tampilkan side-by-side
st.markdown("---")
st.markdown("### 📊 Perbandingan Output")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 📄 Standard")
    st.text_area(
        "Output A",
        value=st.session_state.get("standard_text",""),
        height=350,
        key="display_standard",
    )
    score_a = st.session_state.get("judge_score_before", 0)
    trig_a  = st.session_state.get("standard_trigram", 0)
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.metric("Judge Score", f"{score_a:.0f}/100")
    with col_a2:
        color_a = "🟢" if trig_a < 0.15 else "🟡" if trig_a < 0.30 else "🔴"
        st.metric("Trigram", f"{color_a} {trig_a:.0%}")
    
    # Download A
    st.download_button(
        "⬇ Unduh Standard (.txt)",
        data=st.session_state.get("standard_text",""),
        file_name="output_standard.txt",
        mime="text/plain",
        key="dl_standard",
    )

with col_b:
    improved = st.session_state.get("enhanced_improved", False)
    label_b  = "#### ✨ Judge Enhanced" if improved else "#### ✅ Sudah Optimal"
    st.markdown(label_b)

    enhanced_text = st.session_state.get("enhanced_text","")
    st.text_area(
        "Output B",
        value=enhanced_text,
        height=350,
        key="display_enhanced",
    )
    score_b = st.session_state.get("judge_score_after", 0)
    trig_b  = st.session_state.get("enhanced_trigram", 0)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        delta_score = score_b - score_a
        st.metric(
            "Judge Score",
            f"{score_b:.0f}/100",
            delta=f"+{delta_score:.0f}" if delta_score > 0 else f"{delta_score:.0f}",
        )
    with col_b2:
        color_b = "🟢" if trig_b < 0.15 else "🟡" if trig_b < 0.30 else "🔴"
        delta_trig = trig_b - trig_a
        st.metric(
            "Trigram",
            f"{color_b} {trig_b:.0%}",
            delta=f"{delta_trig:+.0%}",
        )

    if not improved:
        st.info("Output sudah optimal — Enhanced sama dengan Standard.")

    if st.session_state.get("worst_sentence"):
        st.caption(
            f"⚠️ Kalimat yang diperbaiki: "
            f"*\"{st.session_state['worst_sentence'][:80]}...\"*"
        )

    st.download_button(
        "⬇ Unduh Enhanced (.txt)",
        data=enhanced_text,
        file_name="output_enhanced.txt",
        mime="text/plain",
        key="dl_enhanced",
    )


═══════════════════════════════════════════════════════
PERUBAHAN 5: Update Tab Evaluasi — GPTZero 2x input
═══════════════════════════════════════════════════════

Di tab Evaluasi (tab2), ganti section GPTZero manual input
dengan versi 2-kolom untuk Standard vs Enhanced:

st.markdown("---")
st.markdown("#### 🎯 Input Skor GPTZero Manual")
st.caption(
    "Paste Output A dan B ke GPTZero Advanced Scan, "
    "lalu input hasilnya di sini untuk perbandingan."
)

gz_col_a, gz_col_b = st.columns(2)

with gz_col_a:
    st.markdown("**📄 Output A — Standard**")
    gz_std_ai    = st.slider("AI %",    0, 100, 50, key="gz_std_ai")
    gz_std_mixed = st.slider("Mixed %", 0, 100, 0,  key="gz_std_mixed")
    gz_std_human = st.slider("Human %", 0, 100, 0,  key="gz_std_human")

with gz_col_b:
    st.markdown("**✨ Output B — Enhanced**")
    gz_enh_ai    = st.slider("AI %",    0, 100, 50, key="gz_enh_ai")
    gz_enh_mixed = st.slider("Mixed %", 0, 100, 0,  key="gz_enh_mixed")
    gz_enh_human = st.slider("Human %", 0, 100, 0,  key="gz_enh_human")

if st.button("💾 Simpan Perbandingan GPTZero", type="primary"):
    record_id = st.session_state.get("last_record_id")
    if record_id:
        patch_resp = requests.patch(
            f"{API_URL}/api/v1/evaluate/gptzero",
            json={
                "record_id":      record_id,
                "gptzero_std_ai":    gz_std_ai,
                "gptzero_std_mixed": gz_std_mixed,
                "gptzero_std_human": gz_std_human,
                "gptzero_enh_ai":    gz_enh_ai,
                "gptzero_enh_mixed": gz_enh_mixed,
                "gptzero_enh_human": gz_enh_human,
            }
        )
        if patch_resp.status_code == 200:
            # Tampilkan delta
            delta_ai    = gz_enh_ai - gz_std_ai
            delta_human = gz_enh_human - gz_std_human
            
            if delta_ai < 0:
                st.success(
                    f"✅ Enhanced lebih baik! "
                    f"AI% turun {abs(delta_ai)}% | "
                    f"Human% naik {delta_human}%"
                )
            elif delta_ai == 0:
                st.info("Kedua output mendapat skor GPTZero yang sama.")
            else:
                st.warning(
                    f"⚠️ Standard lebih baik dari Enhanced "
                    f"({delta_ai:+d}% AI)"
                )
        else:
            st.error("Gagal menyimpan. Pastikan sudah menjalankan humanize.")
    else:
        st.error("Jalankan Humanize dulu sebelum input skor GPTZero.")


═══════════════════════════════════════════════════════
Setelah semua perubahan:
1. Jalankan pytest
2. Start backend: uvicorn app.main:app --reload
3. Test flow lengkap:
   a. Paste teks di tab Humanize
   b. Klik Humanize → tunggu Output A streaming selesai
   c. Tunggu Output B muncul (Judge Enhancement)
   d. Bandingkan Judge Score dan Trigram keduanya
   e. Buka GPTZero → test kedua output → input skor
   f. Klik Simpan Perbandingan
4. Cek tab History — data A vs B harus tersimpan
═══════════════════════════════════════════════════════
"""HumanWrite AI — Enhanced Streamlit GUI (Global Pre-trained Mode).

Provides a premium interface for humanizing LLM-generated drafts using a
pre-trained global style profile.
"""

import difflib
import os
import re
import time
import json

import requests
import streamlit as st
import pandas as pd

# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HumanWrite AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Look ───────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #a0a0b0;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #667eea;
    }

    /* Diff view */
    .diff-add {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .diff-remove {
        background-color: rgba(248, 81, 73, 0.15);
        color: #f85149;
        text-decoration: line-through;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .diff-container {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        overflow-x: auto;
    }

    /* Score gauge */
    .gauge-container {
        text-align: center;
        padding: 1rem;
    }
    .gauge-score {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .gauge-label {
        font-size: 0.85rem;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online { background-color: #3fb950; }
    .status-offline { background-color: #f85149; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }

    /* Changes list */
    .change-item {
        background: rgba(102, 126, 234, 0.08);
        border-left: 3px solid #667eea;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Configuration ─────────────────────────────────────────────────────────────

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("API_KEY", "humanwrite-secret-key")
HEADERS = {"X-API-Key": API_KEY}


def check_backend() -> bool:
    """Check if the backend is running."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def get_global_style() -> dict:
    """Fetch the pre-trained global style profile."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/v1/style", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

def process_stream(response, thought_placeholder=None, info_placeholder=None):
    """Generator to process the SSE streaming response from the backend."""
    current_event = None
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
            
        if line.startswith("event: "):
            current_event = line[len("event: "):].strip()
        elif line.startswith("data: "):
            data_str = line[len("data: "):]
            if current_event == "text":
                try:
                    chunk = json.loads(data_str)
                    if "<!-- turnitin_refine -->" in chunk:
                        if info_placeholder:
                            info_placeholder.info("🔄 Overlap struktural Turnitin tinggi (> 30%). Memulai Pass-2 (Perombakan Struktural)...")
                        continue
                    if "<thought>" in chunk:
                        if thought_placeholder:
                            with thought_placeholder.expander("🤔 Agent's Chain of Thought (Perencanaan)", expanded=False):
                                st.markdown(f"```text\n{chunk.replace('<thought>', '').replace('</thought>', '').strip()}\n```")
                        continue
                    yield chunk
                except Exception as e:
                    st.error(f"Error parsing stream: {e}")
            elif current_event == "metrics":
                try:
                    st.session_state["last_metrics"] = json.loads(data_str)
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing metrics: {e}")

def render_diff(old_text: str, new_text: str) -> str:
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile="Draf Asli",
        tofile="Teks Hasil Humanize",
        lineterm="",
    )
    diff_lines = list(diff)
    if not diff_lines:
        return '<p style="color:#a0a0b0;">Tidak ada perbedaan signifikan yang terdeteksi.</p>'
    html_parts = []
    for line in diff_lines:
        line_escaped = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if line.startswith("+") and not line.startswith("+++"):
            html_parts.append(f'<span class="diff-add">{line_escaped}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            html_parts.append(f'<span class="diff-remove">{line_escaped}</span>')
        elif line.startswith("@@"):
            html_parts.append(f'<span style="color:#8b949e;">{line_escaped}</span>')
        else:
            html_parts.append(line_escaped)
    return f'<div class="diff-container">{"<br>".join(html_parts)}</div>'

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>✍️ HumanWrite AI</h1>
    <p>Voice Preservation Engine — Pre-trained Global Human Style</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ System Status")

    # Backend status
    backend_online = check_backend()
    status_class = "status-online" if backend_online else "status-offline"
    status_text = "Backend Online" if backend_online else "Backend Offline"
    st.markdown(
        f'<span class="status-dot {status_class}"></span> {status_text}',
        unsafe_allow_html=True,
    )

    if not backend_online:
        st.warning("⚠️ Backend is not running.")
        
    st.markdown("---")
    st.markdown(
        "<small style='color: #666;'>HumanWrite AI v0.2.0<br>"
        "Powered by Groq LLaMA 3.3-70b</small>",
        unsafe_allow_html=True,
    )


# ── Main Content ──────────────────────────────────────────────────────────────

profile = get_global_style() if backend_online else {}

if profile:
    with st.expander("📊 Lihat Metrik Profil Global", expanded=False):
        st.markdown(f"**Bahasa Utama:** `{profile.get('language', 'id').upper()}`")
        
        # Metric cards
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Rata-rata Kata/Kalimat</div><div class="metric-value">{profile.get('avg_sentence_length', 0):.1f}</div></div>""", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Readability (Flesch)</div><div class="metric-value">{profile.get('flesch_reading_ease', 0):.1f}</div></div>""", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Kalimat Aktif</div><div class="metric-value">{profile.get('active_voice_ratio', 0):.0%}</div></div>""", unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Register</div><div class="metric-value" style="font-size:1.2rem;">{profile.get('emotion_register', 'neutral').title()}</div></div>""", unsafe_allow_html=True)

if not profile and backend_online:
    st.error("Global Style Profile tidak ditemukan. Harap jalankan script training di backend.")

tab1, tab2 = st.tabs([
    "✍️ Humanize", 
    "📜 History"
])

# ── Tab 1: Humanize ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Masukkan Teks AI")
    st.caption("Tempel draf yang dihasilkan AI di sini untuk ditulis ulang menjadi lebih natural.")

    style_mode_options = ["populer", "akademik", "profesional", "kreatif"]
    style_mode_val = st.selectbox(
        "Pilih Mode Gaya Penulisan",
        options=style_mode_options,
        index=0,
    )

    draft_text = st.text_area(
        "Teks Draf",
        height=250,
        placeholder="Tempel draf AI Anda di sini...",
        label_visibility="collapsed",
    )

    humanize_btn = st.button(
        "🪄 Humanize Teks",
        use_container_width=True,
        disabled=not draft_text or not backend_online or not profile,
    )

    if humanize_btn and draft_text:
        with st.spinner("✨ Sedang menulis ulang dengan gaya manusia..."):
            try:
                process_resp = requests.post(
                    f"{BACKEND_URL}/api/v1/process",
                    json={"draft": draft_text, "style_mode": style_mode_val},
                    headers=HEADERS,
                    stream=True,
                    timeout=120,
                )
                process_resp.raise_for_status()
                
                st.markdown("### ✅ Hasil Humanize")
                st.session_state["last_metrics"] = None
                
                info_placeholder = st.empty()
                thought_placeholder = st.empty()
                text_placeholder = st.empty()
                
                # Consume the stream
                final_text = text_placeholder.write_stream(process_stream(process_resp, thought_placeholder, info_placeholder))
                
                st.session_state["last_result"] = final_text
                st.session_state["last_draft"] = draft_text
                st.session_state["style_mode_val"] = style_mode_val
                
                # Log the evaluation to get a record_id
                eval_resp = requests.post(
                    f"{BACKEND_URL}/api/v1/evaluate/run",
                    json={
                        "style_mode": style_mode_val,
                        "language": profile.get("language", "id"),
                        "original_text": draft_text,
                        "output_text": final_text,
                    },
                    headers=HEADERS,
                    timeout=10,
                )
                if eval_resp.status_code == 200:
                    st.session_state["last_record_id"] = eval_resp.json().get("record_id")


                        
            except Exception as e:
                st.error(f"Proses gagal: {e}")

    # Display results
    if st.session_state.get("last_result"):
        final_text = st.session_state["last_result"]
        original_draft = st.session_state.get("last_draft", "")
        metrics = st.session_state.get("last_metrics") or {}
        changes = metrics.get("changes_made", [])
        st.markdown("---")
        
        # Word count
        word_count_orig = len(original_draft.split())
        st.caption(f"**Jumlah Kata Asli:** {word_count_orig}")

        st.markdown(f"**Jumlah Kata Hasil:** {len(final_text.split())}")
        st.text_area("Hasil Output", final_text, height=300, disabled=True)
        
        st.download_button(
            label="📄 Download Output (.txt)",
            data=final_text,
            file_name=f"output_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        with st.expander("🔀 Tampilan Diff (Output vs Asli)", expanded=False):
            st.markdown(render_diff(original_draft, final_text), unsafe_allow_html=True)
            
        if changes:
            with st.expander("📋 Perubahan yang Dilakukan", expanded=False):
                for change in changes:
                    st.markdown(f'<div class="change-item">💡 {change}</div>', unsafe_allow_html=True)

        # ── Evaluasi (Dipindah dari Tab 2) ────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🛡️ Turnitin Safety & Anti-Deteksi")

        if "last_record_id" in st.session_state:
            # Hitung overlap lokal jika endpoint evaluasi belum menghitung
            try:
                from app.core.text_utils import check_trigram_overlap
                trigram_overlap = check_trigram_overlap(original_draft, final_text)
            except Exception:
                trigram_overlap = metrics.get("trigram_overlap", 0) if metrics else 0

            overlap_pct = int(trigram_overlap * 100)

            if overlap_pct < 15:
                st.success(f"✅ **{overlap_pct}% kesamaan struktural** — Aman untuk Turnitin")
            elif overlap_pct < 30:
                st.warning(f"⚠️ **{overlap_pct}% kesamaan struktural** — Perlu perhatian")
            else:
                st.error(f"🔴 **{overlap_pct}% kesamaan struktural** — Risiko tinggi plagiarisme")

            # Progress bar visual
            st.progress(
                min(overlap_pct / 100, 1.0),
                text=f"Turnitin overlap: {overlap_pct}%"
            )

            st.caption(
                "🟢 < 15%: Aman  |  "
                "🟡 15-30%: Perlu perhatian  |  "
                "🔴 > 30%: Risiko tinggi"
            )
        else:
            st.info("Humanize teks terlebih dahulu untuk melihat Turnitin Safety Score.")
                    


# ── Tab 2: History ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📜 Dasbor & Riwayat Evaluasi")
    st.caption("Analitik performa HumanWrite AI dan daftar historis evaluasi.")
    
    if st.button("🔄 Segarkan Data & Dasbor"):
        try:
            resp = requests.get(f"{BACKEND_URL}/api/v1/evaluate/history", headers=HEADERS)
            if resp.status_code == 200:
                history_data = resp.json().get("data", [])
                if history_data:
                    df = pd.DataFrame(history_data)
                    
                    # Dashboard Analytics
                    st.markdown("#### 📈 Tren Skor GPTZero (Human %)")
                    if "gptzero_enh_human" in df.columns:
                        chart_data = df[df["gptzero_enh_human"].notna()].sort_values("id")
                        if not chart_data.empty:
                            st.line_chart(chart_data.set_index("id")["gptzero_enh_human"], color="#3fb950")
                        else:
                            st.caption("Belum ada data GPTZero manual.")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if "trigram_overlap" in df.columns:
                            avg_overlap = df["trigram_overlap"].mean() * 100
                            st.metric("Rata-rata Turnitin Overlap", f"{avg_overlap:.1f}%")
                    with col2:
                        if "enhanced_judge_score" in df.columns:
                            avg_score = df["enhanced_judge_score"].mean()
                            st.metric("Rata-rata Skor LLM Judge", f"{avg_score:.1f}/100")
                    with col3:
                        st.metric("Total Dokumen", f"{len(df)}")
                        
                    st.markdown("---")
                    st.markdown("#### Tabel Riwayat Detail")
                    cols_to_show = ["id", "timestamp", "style_mode", "enhanced_judge_score", "gptzero_enh_human", "trigram_overlap"]
                    existing_cols = [c for c in cols_to_show if c in df.columns]
                    st.dataframe(df[existing_cols].sort_values(by="id", ascending=False), use_container_width=True)
                else:
                    st.info("Belum ada data evaluasi di database.")
        except Exception as e:
            st.error(f"Gagal mengambil riwayat: {e}")
    else:
        st.info("Klik tombol di atas untuk memuat dasbor dan riwayat terbaru.")

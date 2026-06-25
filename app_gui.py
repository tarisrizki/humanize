"""HumanWrite AI — Enhanced Streamlit GUI (Global Pre-trained Mode).

Provides a premium interface for humanizing LLM-generated drafts using a
pre-trained global style profile.
"""

import difflib
import os
import io
import time

import requests
import streamlit as st

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
        "Powered by Gemini 2.5 Flash</small>",
        unsafe_allow_html=True,
    )


# ── Main Content ──────────────────────────────────────────────────────────────

profile = get_global_style() if backend_online else {}

if profile:
    with st.expander("📊 View Global Style Profile Metrics", expanded=False):
        st.markdown(f"**Primary Language:** `{profile.get('language', 'en').upper()}`")
        
        # Metric cards
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg Sentence Length</div><div class="metric-value">{profile.get('avg_sentence_length', 0):.1f}</div></div>""", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Readability (Flesch)</div><div class="metric-value">{profile.get('flesch_reading_ease', 0):.1f}</div></div>""", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Active Voice</div><div class="metric-value">{profile.get('active_voice_ratio', 0):.0%}</div></div>""", unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Register</div><div class="metric-value" style="font-size:1.2rem;">{profile.get('emotion_register', 'neutral').title()}</div></div>""", unsafe_allow_html=True)

st.markdown("### Paste Your LLM-Generated Draft")
st.caption("Enter the AI-generated text you want to rewrite using our pre-trained Human Style.")

style_mode_options = ["populer", "akademik", "profesional", "kreatif"]
style_mode_val = st.selectbox(
    "Select Style Mode",
    options=style_mode_options,
    index=0,
)

draft_text = st.text_area(
    "Draft text",
    height=250,
    placeholder="Paste your AI-generated draft here...",
    label_visibility="collapsed",
)

humanize_btn = st.button(
    "🪄 Humanize Draft",
    use_container_width=True,
    disabled=not draft_text or not backend_online or not profile,
)

if not profile and backend_online:
    st.error("Global Style Profile not found. Please run the training script on the backend.")

import json

def process_stream(response):
    """Generator to process the SSE streaming response from the backend."""
    # We can use a simple manual SSE parser since we only expect 'event' and 'data' lines
    # Response.iter_lines is safe for this.
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
                    # We sent json-encoded chunks to be safe with newlines
                    chunk = json.loads(data_str)
                    yield chunk
                except:
                    pass
            elif current_event == "metrics":
                try:
                    st.session_state["last_metrics"] = json.loads(data_str)
                except json.JSONDecodeError:
                    pass

if humanize_btn and draft_text:
    with st.spinner("✨ Rewriting with human voice..."):
        try:
            process_resp = requests.post(
                f"{BACKEND_URL}/api/v1/process",
                json={"draft": draft_text, "style_mode": style_mode_val},
                stream=True,
                timeout=120,
            )
            process_resp.raise_for_status()
            
            st.markdown("### ✅ Humanized Result")
            st.session_state["last_metrics"] = None
            
            # Create a placeholder for the streamed text
            text_placeholder = st.empty()
            
            # Consume the stream
            final_text = text_placeholder.write_stream(process_stream(process_resp))
            
            st.session_state["last_result"] = final_text
            st.session_state["last_draft"] = draft_text
            
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.stop()

# Display results
if st.session_state.get("last_result"):
    final_text = st.session_state["last_result"]
    original_draft = st.session_state.get("last_draft", "")
    metrics = st.session_state.get("last_metrics") or {}
    changes = metrics.get("changes_made", [])

    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Changes Made")
        if changes:
            for change in changes:
                st.markdown(f"- {change}")
        else:
            st.info("No major changes documented.")
            
    with col2:
        st.subheader("Original Draft")
        st.info(original_draft)



    # ── Side by side: Original vs Final ──────────────────────────
    st.markdown("#### 📝 Comparison")
    left, right = st.columns(2)

    with left:
        st.markdown("**Original Draft**")
        st.text_area(
            "Original",
            value=original_draft,
            height=200,
            disabled=True,
            label_visibility="collapsed",
        )

    with right:
        st.markdown("**Humanized Text**")
        st.text_area(
            "Final",
            value=final_text,
            height=200,
            disabled=True,
            label_visibility="collapsed",
        )

    # ── Diff View ─────────────────────────────────────────────────
    with st.expander("🔀 Diff View (line-by-line changes)", expanded=True):
        diff = difflib.unified_diff(
            original_draft.splitlines(keepends=True),
            final_text.splitlines(keepends=True),
            fromfile="Original Draft",
            tofile="Humanized Text",
            lineterm="",
        )
        diff_lines = list(diff)

        if diff_lines:
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

            diff_html = "<br>".join(html_parts)
            st.markdown(
                f'<div class="diff-container">{diff_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No differences detected — the text may already match your style.")

    # ── Changes List ──────────────────────────────────────────────
    if changes:
        st.markdown("#### 📋 Changes Made")
        for i, change in enumerate(changes, 1):
            st.markdown(
                f'<div class="change-item">💡 {change}</div>',
                unsafe_allow_html=True,
            )

    # ── Export Buttons ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📥 Export")
    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        st.download_button(
            label="📄 Download as .txt",
            data=final_text,
            file_name=f"humanized_global_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with export_col2:
        try:
            from docx import Document as DocxDocument

            doc = DocxDocument()
            doc.add_heading("HumanWrite AI — Humanized Text", level=1)
            doc.add_paragraph("Style: Global Pre-trained Model")
            doc.add_heading("Humanized Text", level=2)
            for para in final_text.split("\n\n"):
                doc.add_paragraph(para)

            buffer = io.BytesIO()
            doc.save(buffer)
            buffer.seek(0)

            st.download_button(
                label="📝 Download as .docx",
                data=buffer,
                file_name=f"humanized_global_{int(time.time())}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except ImportError:
            st.button("📝 .docx (install python-docx)", disabled=True, use_container_width=True)

    with export_col3:
        # Copy to clipboard via text area trick
        st.text_area(
            "Copy text",
            value=final_text,
            height=68,
            help="Select all and copy (Ctrl+A, Ctrl+C)",
            label_visibility="collapsed",
        )

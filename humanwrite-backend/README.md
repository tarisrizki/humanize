# ✍️ HumanWrite AI

**Voice Preservation Engine** — Rewrite LLM-generated drafts to match your personal writing style.

## How It Works

1. **Upload** 1–3 documents that represent your writing style (TXT, DOCX, PDF)
2. **Analyze** — The system extracts a quantitative Style Profile from your writing
3. **Humanize** — Paste any AI-generated draft and get it rewritten in *your* voice

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend API | FastAPI (Python 3.11+) |
| NLP (English) | spaCy (`en_core_web_sm`) |
| NLP (Indonesian) | Stanza (`id`) |
| Readability | textstat |
| AI Rewriting | PydanticAI + Gemini 2.5 Flash |
| GUI | Streamlit |
| Package Manager | uv |

## Quick Start

### 1. Install Dependencies

```bash
cd humanwrite-backend
uv sync
uv sync --extra gui
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=your_actual_key_here
```

### 3. Start the Backend

```bash
cd humanwrite-backend
uv run uvicorn app.main:app --reload --port 8000
```

The first run will automatically download spaCy and Stanza models.

### 4. Start the GUI (separate terminal)

```bash
cd humanwrite-backend
uv run streamlit run ../app_gui.py --server.port 8501
```

### 5. Open the App

Navigate to [http://localhost:8501](http://localhost:8501) in your browser.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check & model status |
| `POST` | `/api/v1/upload` | Upload documents (multipart form) |
| `POST` | `/api/v1/analyze` | Analyze writing style from corpus |
| `POST` | `/api/v1/process` | Rewrite draft with user's style |

## Style Profile Metrics

The analyzer extracts these metrics from your writing:

- **avg_sentence_length** — Average words per sentence
- **sentence_length_std** — Variation in sentence lengths
- **flesch_reading_ease** — Readability score (0–100)
- **active_voice_ratio** — Active vs passive voice usage
- **top_transitions** — Most used transition words
- **hedging_frequency** — Cautious language frequency
- **emotion_register** — Formal / informal / neutral
- **favorite_phrases** — Recurring multi-word expressions
- **lexical_diversity** — Vocabulary variety (Type-Token Ratio)
- **paragraph_length_avg** — Average sentences per paragraph
- **language** — Auto-detected: English (`en`) or Indonesian (`id`)

## Project Structure

```
write/
├── app_gui.py                          # Streamlit GUI
├── humanwrite-backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI entry point
│   │   ├── config.py                   # Environment configuration
│   │   ├── api/v1/endpoints/
│   │   │   ├── upload.py               # Document upload
│   │   │   ├── analyze.py              # Style analysis
│   │   │   └── process.py              # Draft processing
│   │   ├── core/
│   │   │   ├── style_analyzer.py       # NLP analysis (spaCy + Stanza)
│   │   │   └── writing_engine.py       # AI rewriting (PydanticAI)
│   │   ├── models/
│   │   │   ├── style_profile.py        # StyleProfile model
│   │   │   └── schemas.py              # API schemas
│   │   └── storage/
│   │       └── json_store.py           # JSON file storage
│   ├── data/                           # Runtime data (auto-created)
│   │   ├── corpus/                     # Uploaded text files
│   │   ├── profiles/                   # Generated style profiles
│   │   └── outputs/                    # Processing results
│   ├── pyproject.toml
│   ├── .env.example
│   └── README.md
└── p.md                                # Original spec
```

## License

Private project.

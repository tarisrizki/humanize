# ✍️ HumanWrite AI

**Voice Preservation Engine** — Rewrite LLM-generated drafts to match your personal writing style.

## How It Works

1. **Train** — The system analyzes a corpus of 8,000+ texts (Wikipedia, News, Papers, Fiction) to extract quantitative Style Profiles using Stanza & spaCy.
2. **Humanize** — Paste any AI-generated draft, select your desired style mode (Akademik, Profesional, Populer, Kreatif) and get it rewritten with natural human burstiness.
3. **Evaluate** — Built-in LLM Judge automatically scores the output across 5 dimensions, and GPTZero scores can be manually tracked in the database.

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
| Dataset Extraction | HuggingFace Datasets & Pandas |

## Key Features
- **Bilingual Support**: Fully supports English and Indonesian with language-specific Flesch scoring and NLP engines.
- **Language-Aware Few-Shots**: Automatically detects draft language (via `langdetect`) and injects matching reference texts (English few-shots for English drafts, Indonesian for Indonesian).
- **CSV Data Architecture**: Uses single-file CSVs for large 8,000+ document datasets rather than splitting into thousands of tiny `.txt` files, making Git operations instantaneous.

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
| `GET` | `/api/v1/style` | Get the currently active global Style Profile |
| `POST` | `/api/v1/process` | Rewrite draft using streaming Server-Sent Events (SSE) |
| `POST` | `/api/v1/evaluate/run` | Log a humanize run to the DB |
| `POST` | `/api/v1/evaluate/judge` | Run the Llama-3.3 LLM Judge |
| `PATCH`| `/api/v1/evaluate/{id}/gptzero` | Update GPTZero scores |
| `GET` | `/api/v1/evaluate/history` | Get all past evaluation results |

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
- **few_shot_examples_id** — Top reference texts closest to the mean (Indonesian)
- **few_shot_examples_en** — Top reference texts closest to the mean (English)

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
│   │   ├── corpus_akademik.csv         # 2,000 texts bilingual dataset
│   │   ├── corpus_profesional.csv      # 2,000 texts bilingual dataset
│   │   ├── corpus_populer.csv          # 2,000 texts bilingual dataset
│   │   ├── corpus_kreatif.csv          # 1,000+ texts bilingual dataset
│   │   ├── profiles/                   # Generated JSON style profiles
│   │   └── outputs/                    # Processing results
│   ├── pyproject.toml
│   ├── .env.example
│   └── README.md
└── p.md                                # Original spec
```

## License

Private project.

"""FastAPI application entry point.

Loads spaCy (English) and Stanza (Indonesian) models on startup,
configures CORS, and includes all API routers.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core import style_analyzer

logger = logging.getLogger("humanwrite")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load NLP models on startup, clean up on shutdown."""

    # ── Load spaCy English model ──────────────────────────────────────────
    logger.info("Loading spaCy English model: %s", settings.SPACY_MODEL_EN)
    try:
        import spacy

        try:
            nlp_en = spacy.load(settings.SPACY_MODEL_EN)
        except OSError:
            logger.info("Downloading spaCy model '%s'...", settings.SPACY_MODEL_EN)
            spacy.cli.download(settings.SPACY_MODEL_EN)
            nlp_en = spacy.load(settings.SPACY_MODEL_EN)

        style_analyzer.register_spacy(nlp_en)
        logger.info("spaCy English model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load spaCy English model: %s", e)

    # ── Load Stanza Indonesian model ──────────────────────────────────────
    logger.info("Loading Stanza Indonesian model...")
    try:
        import stanza

        # Download Indonesian model if not present (silent if already downloaded)
        stanza.download("id", verbose=False)

        nlp_id = stanza.Pipeline(
            "id",
            processors="tokenize,pos,lemma,depparse",
            verbose=False,
        )

        style_analyzer.register_stanza(nlp_id)
        logger.info("Stanza Indonesian model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load Stanza Indonesian model: %s", e)
        logger.warning("Indonesian language support will be unavailable.")

    yield  # Application runs

    # Cleanup (nothing to do for now)
    logger.info("Shutting down HumanWrite AI.")


# ── Create the FastAPI application ────────────────────────────────────────────

app = FastAPI(
    title="HumanWrite AI",
    description="Rewrite LLM drafts to match your personal writing style.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "service": "HumanWrite AI",
        "spacy_loaded": style_analyzer._spacy_nlp is not None,
        "stanza_loaded": style_analyzer._stanza_pipeline is not None,
    }


# ── Include API routers ──────────────────────────────────────────────────────

from app.api.v1.endpoints import process

app.include_router(process.router, prefix="/api/v1", tags=["process"])

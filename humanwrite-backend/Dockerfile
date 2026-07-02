FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (build-essential needed for some C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install uv

# Copy uv lock files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv pip install --system -r pyproject.toml

# Pre-download NLP models to be cached in the image
RUN python -m spacy download en_core_web_sm
RUN python -c "import stanza; stanza.download('id')"

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

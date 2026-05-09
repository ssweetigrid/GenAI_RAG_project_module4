# ── Base image ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Set working directory ─────────────────────────────────────────────────
WORKDIR /app

# ── Install system dependencies ───────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first (for Docker layer caching) ────────────────────
COPY requirements.txt .

# ── Install Python packages ───────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy the entire project ───────────────────────────────────────────────
COPY . .

# ── Create data directories ───────────────────────────────────────────────
RUN mkdir -p data/raw data/processed/text data/processed/tables \
             data/processed/images keys

# ── Expose Streamlit port ─────────────────────────────────────────────────
EXPOSE 8501

# ── Environment: tell Streamlit not to open browser ───────────────────────
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# ── Start Streamlit UI ─────────────────────────────────────────────────────
CMD ["streamlit", "run", "src/ui/streamlit_app.py", "--server.port=8501"]

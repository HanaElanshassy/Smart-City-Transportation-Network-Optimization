# ─────────────────────────────────────────────────────────────
# Cairo Smart City — Production Dockerfile
# Base: Python 3.11 slim
# Fixed: libgl1-mesa-glx → libgl1 (renamed in Debian Trixie/13)
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps needed by matplotlib / scikit-learn
# libgl1 replaces libgl1-mesa-glx in Debian 12+ / Trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory inside the container
WORKDIR /app

# Copy requirements first — Docker caches this layer separately
# so a code-only change doesn't re-install all packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Streamlit config — disable telemetry, set server options
RUN mkdir -p /root/.streamlit
RUN printf '\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
base = "dark"\n\
' > /root/.streamlit/config.toml

# Render uses PORT env var — this default is overridden automatically
EXPOSE 8501

# Entrypoint
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]

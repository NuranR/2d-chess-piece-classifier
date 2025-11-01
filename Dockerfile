FROM python:3.9-slim

# Avoid interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install small set of system deps required by OpenCV and for healthchecks
# Use --no-install-recommends to keep image small and combine into single layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements first to leverage Docker layer caching when possible
COPY requirements.txt ./requirements.txt

# Upgrade pip and install Python dependencies (no-cache-dir to reduce size)
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy only application code and models after deps are installed
COPY . .

# Create .streamlit directory (if the repo doesn't already include it)
RUN mkdir -p .streamlit || true

# Expose Streamlit default port
EXPOSE 8501

# Health check (curl is provided above)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app in production mode
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]

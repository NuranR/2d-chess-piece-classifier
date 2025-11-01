FROM continuumio/miniconda3

# Avoid interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

COPY environment.yml ./environment.yml

# Create the Conda environment from envtfile
RUN conda env create -f environment.yml

# 6. Copy the rest of the app code (app.py, models/, etc.)
COPY . .

# 7. Create .streamlit directory 
RUN mkdir -p .streamlit || true

# 8. Expose Streamlit port
EXPOSE 8501

# 9. Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app using the 'conda run' wrapper
# Executes the command inside the 'chess2fen' environment.
CMD ["conda", "run", "-n", "chess2fen", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
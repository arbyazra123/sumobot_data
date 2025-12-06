# ---- Base image ----
FROM python:3.10.16-slim

# ---- Set working directory ----
WORKDIR /app

# ---- Copy project files ----
COPY . /app

# ---- Install dependencies ----
RUN pip install -r requirements.txt

# ---- Expose Streamlit default port ----
EXPOSE 8501

# ---- Set environment for Streamlit ----
ENV STREAMLIT_HOME=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- Run Streamlit ----
CMD ["streamlit", "run", "plotting/streamlit_plot.py", "--server.port=8501", "--server.address=0.0.0.0"]

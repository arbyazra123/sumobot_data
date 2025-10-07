# ---- Base image ----
FROM python:3.10-slim

# ---- Set working directory ----
WORKDIR /app

# ---- Copy project files ----
COPY . /app

# ---- Install dependencies ----
# If you have requirements.txt
RUN pip install -r requirements.txt \
    || pip install streamlit pandas numpy matplotlib seaborn

# ---- Expose Streamlit default port ----
EXPOSE 8501

# ---- Set environment for Streamlit ----
ENV STREAMLIT_HOME=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- Run Streamlit ----
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

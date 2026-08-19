FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for pdf2image
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache utilization
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Keep the upload cap on the image rather than relying on the working
# directory finding .streamlit/config.toml, or on a task-definition
# override that lives outside this repo.
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["src/main_app.py"]
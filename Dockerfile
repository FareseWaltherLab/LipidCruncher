FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for pdf2image
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged user the app runs as (see USER below).
RUN useradd --create-home --uid 1000 app

# Copy requirements first for better cache utilization
COPY requirements.txt constraints.txt .
RUN pip install -r requirements.txt -c constraints.txt

# Copy the rest of the application
COPY . .

# Keep the upload cap on the image rather than relying on the working
# directory finding .streamlit/config.toml, or on a task-definition
# override that lives outside this repo.
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024

EXPOSE 8501

# Never run as root: a compromise of the app would otherwise have root in
# the container.
USER app

ENTRYPOINT ["streamlit", "run"]
CMD ["src/main_app.py"]
FROM python:3.9-slim
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

# Create symbolic links for the images to ensure they're found in all expected locations
RUN mkdir -p /app/src/images
RUN cp -r /app/images/* /app/src/images/ || true

# Expose the Streamlit port
EXPOSE 8501

# Set the entrypoint and command
ENTRYPOINT ["streamlit", "run"]
CMD ["src/main_app.py"]
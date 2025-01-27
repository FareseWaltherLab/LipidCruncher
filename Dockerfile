FROM python:3.9-slim
WORKDIR /app

# Copy requirements first for better cache utilization
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Set the same entrypoint and command as your current image
ENTRYPOINT ["streamlit", "run"]
CMD ["main_app.py"]
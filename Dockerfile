# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY professional_analyzer.py .
COPY vision_api_manager.py .
COPY web_app.py .

# Create directories for templates and static (will be created by app)
RUN mkdir -p templates static

# Expose port
EXPOSE 5555

# Environment variables (can be overridden)
ENV FLASK_APP=web_app.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "web_app.py"]
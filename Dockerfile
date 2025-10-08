# Use Python 3.10 slim image as base (Render only supports Python <= 3.11)
FROM python:3.10-slim

# Install system dependencies required for OpenCV, MediaPipe, and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to comply with Render requirements
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt to the container and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that the app will run on (Render uses port 10000 by default)
EXPOSE 10000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]

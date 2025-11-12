# Use a Python base image with a Debian distribution (Bookworm) 
# to ensure compatibility with libraries like Faiss, which requires system packages.
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install system dependencies required by Faiss, NumPy, and Scikit-Surprise
# Faiss often requires an optimized linear algebra library like OpenBLAS.
# We also install 'gcc' and 'gfortran' which are often needed to compile the C/C++ components 
# of the ML libraries during the 'pip install' process.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

# 1. Install Python dependencies
# Copy requirements file first to take advantage of Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy application code
# The entire src directory is needed
COPY src ./src

# Create the models directory where artifacts are expected to be saved
# Note: This directory will later be mounted as a volume in docker-compose.
RUN mkdir -p /app/models

# 3. Expose port and define startup command
# FastAPI runs on port 8000 as defined in your README
EXPOSE 8000

# Command to run the application using Uvicorn.
# We specify the module (src.recommender.api) and the FastAPI instance (app).
# --host 0.0.0.0 is mandatory for Docker to expose the port correctly.
CMD ["uvicorn", "src.recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]
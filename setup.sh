#!/bin/bash

# --- Script to initialize the FrameRates project structure ---

PROJECT_NAME="FrameRate"
CORE_PACKAGE="recommender"

echo "Initializing project structure for $PROJECT_NAME..."

# 1. Create top-level directories
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

echo "Creating core directories..."
mkdir -p data/raw data/processed
mkdir -p models
mkdir -p src/$CORE_PACKAGE
mkdir -p notebooks
mkdir -p tests

# 2. Create crucial empty files and placeholders
echo "Creating essential files..."

# Core Python package marker
touch src/$CORE_PACKAGE/__init__.py

# Main API and training scripts (empty stubs)
touch src/$CORE_PACKAGE/api.py
touch src/$CORE_PACKAGE/pipeline_train.py
touch src/$CORE_PACKAGE/streamlit_ui.py
touch src/$CORE_PACKAGE/utils.py

# Placeholder files for Git (so the directories are tracked even if empty)
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch tests/.gitkeep

# 3. Create .gitignore to prevent committing large files
echo "Creating .gitignore..."
cat > .gitignore << EOL
# Ignore environments and cache
.venv/
__pycache__/
*.pyc
*.swp

# Ignore data and model artifacts
data/
models/
!models/index.faiss  # Keep the model artifact, but ignore local run logs
!models/svd_model.pkl

# Ignore logs and documentation
*.log
.DS_Store
EOL

# 4. Create the requirements.txt file (Initial dependencies)
echo "Creating requirements.txt..."
cat > requirements.txt << EOL
# Core Data Science
pandas
numpy
scikit-learn
scikit-surprise # For Matrix Factorization/SVD

# High-Performance Search
faiss-cpu # Use faiss-gpu if you have a compatible GPU

# MLOps & Serving
mlflow
fastapi
uvicorn[standard]
pydantic
streamlit

# Development
pytest
requests
EOL

# 5. Create Dockerfile (Base for the FastAPI server)
echo "Creating Dockerfile..."
cat > Dockerfile << EOL
# Use a Python base image optimized for performance
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies (Faiss may require specific libraries in production)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY src /app/src
COPY models /app/models
COPY data /app/data

# Command to run the FastAPI application using Uvicorn
# Assumes your FastAPI app instance is named 'app' in 'src.recommender.api'
CMD ["uvicorn", "src.recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]
EOL

# 6. Create a basic docker-compose.yml (Orchestration)
echo "Creating docker-compose.yml..."
cat > docker-compose.yml << EOL
version: '3.8'

services:
  # 1. FastAPI Model Serving API
  api_service:
    build: .
    container_name: recommender_api
    ports:
      - "8000:8000"
    volumes:
      # Mount the source code for easier local development
      - ./src:/app/src
      - ./models:/app/models
      - ./data:/app/data
    restart: always

  # 2. Streamlit UI (Client for the API)
  streamlit_ui:
    build: .
    container_name: recommender_ui
    command: streamlit run src/recommender/streamlit_ui.py
    ports:
      - "8501:8501"
    volumes:
      - ./src:/app/src
    environment:
      # API URL Streamlit must connect to
      - API_URL=http://api_service:8000
    depends_on:
      - api_service
    restart: always

  # 3. MLflow Tracking Server
  mlflow_tracker:
    image: mlflow/mlflow
    container_name: mlflow_server
    ports:
      - "5000:5000"
    volumes:
      # Stores artifacts and run data persistently
      - ./mlruns:/mlruns
    environment:
      - MLFLOW_TRACKING_URI=http://0.0.0.0:5000
    command: mlflow server --backend-store-uri file:/mlruns --host 0.0.0.0
    restart: always

echo "
Project initialization complete!
Directory: $PROJECT_NAME/

Next Steps:
1. Populate 'src/recommender/pipeline_train.py' to generate embeddings and the Faiss index.
2. Run 'pip install -r requirements.txt' or use Docker to build the image.
3. Run the full system with: 'docker-compose up --build'
"
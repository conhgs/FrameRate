# FrameRate🎬

**FrameRate** is an end-to-end machine learning project built to recommend a user’s top 10 films in under 50 milliseconds. It’s a project inspired by my love of cinema, data, and the art of making machines think fast.   

This project showcases technical depth across the full stack of ML development — from raw data to running containers:
- **Fast inference**: Optimized model pipelines achieving sub-50 ms latency
- **Model design & evaluation**: Iterative development with robust validation
- **MLOps & experiment tracking**: Full reproducibility and performance monitoring
- **API architecture**: Serving through FastAPI for speed and scalability
- **Deployment**: Containerized with Docker for consistent production delivery
- **Frontend**: A lightweight UI for exploring personalized film recommendations

## 🚀 Project Architecture

The FrameRate architecture is engineered for extremely fast, sub-50ms inference by separating model training from real-time serving using highly specialized tools.

**Training (SVD)**: We employ Singular Value Decomposition (SVD) to analyze user ratings. SVD is a technique that breaks down the massive user-rating matrix into smaller components, representing each user and each movie as a compact list of numbers, or a vector. These vectors efficiently capture underlying tastes and features.

**Indexing (Faiss)**: To achieve speed, we pre-calculate and index all movie vectors using Faiss (Facebook AI Similarity Search). Faiss is a highly optimized library for rapid similarity search. When a request comes in, the system uses the user's vector to instantly query the Faiss index, retrieving the top 10 closest movie vectors in milliseconds.

**Serving (FastAPI)**: The high-performance recommendation logic is exposed via a FastAPI service, providing a fast and scalable API endpoint.

**MLOps**: MLflow is integrated to track all experiments, ensuring full reproducibility and allowing us to compare the performance (speed and accuracy) of different SVD and Faiss configurations.

## ⚙️ Project Structure

A an overview of the directories in this project:

```
.
├── README.md
├── setup.sh
├── requirements.txt     # Python dependencies
├── Dockerfile           # Defines the container for the API service
├── docker-compose.yml   # Orchestrates all services (API, UI, MLflow)
├── src
│   └── recommender      # Core Python source code for the application
│       ├── __init__.py
│       ├── api.py
│       ├── pipeline_train.py
│       └── utils.py
├── tests                # Unit and integration tests
│   └── frontend
├── data                 # Raw and processed datasets (e.g., MovieLens ratings)
│   ├── raw
│   └── processed
├── models               # Trained model artifacts (Faiss index, SVD model)
├── notebooks            # Jupyter notebooks for exploration and analysis
└── frontend             # Static HTML, CSS, and JS for the UI
    ├── index.html
    ├── style.css
    └── main.js
```

## 🛠️ Installation & Setup

This project is containerized using Docker and Docker Compose for a simple and reproducible setup.

### Prerequisites

- Docker and Docker Compose
- Python 3.11.9+ (for local development outside Docker)
- Access to a shell/terminal (like Bash or Zsh)

### 1. Initialize the Project

First, run the setup script to create the directory structure and necessary files.

```bash
./setup.sh
cd FrameRate
```

### 2. Download Data

To download the raw data required to run the project, run the following shell script:

```bash
./data.sh
```

### 3. Build and Run the System

The entire stack (Training Pipeline, API, Frontend, and MLflow) can be orchestrated with Docker Compose.

```
# Run the training pipeline to generate the model artifacts
python3 src/recommender/pipeline_train.py

# Launch all services in detached mode
docker-compose up --build -d
```

Once running, you can access:

**Frontend UI**: http://localhost:8501

**FastAPI Docs**: http://localhost:8000/docs

**MLflow UI**: http://localhost:5000

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
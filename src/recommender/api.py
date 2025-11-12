import os
import pickle
import numpy as np
import pandas as pd
import faiss

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <<< ADDED IMPORT
from pydantic import BaseModel
from typing import List, Dict, Any

from surprise import SVD
from pathlib import Path

# --- Configuration & Paths ---
MODEL_DIR = Path("models")
SVD_MODEL_PATH = MODEL_DIR / "svd_model.pkl"
FAISS_INDEX_PATH = MODEL_DIR / "faiss_index.bin"
MOVIE_MAP_PATH = MODEL_DIR / "movie_map.pkl"
N_RECOMMENDATIONS = 10  # Number of films to return

# --- In-Memory Artifacts ---
# These will be loaded once at startup for fast inference
svd_model: SVD = None
faiss_index: faiss.Index = None
movie_map: pd.DataFrame = None
user_vector_cache: Dict[int, np.ndarray] = {} # Cache for user vectors


# --- FastAPI App Initialization ---
app = FastAPI(
    title="FrameRate Recommender API",
    description="High-speed film recommendation service using SVD and Faiss for sub-50ms latency.",
    version="1.0.0"
)

# --- ADDED CORS Middleware Configuration ---
origins = [
    "http://localhost",
    "http://localhost:8501",  # This is the key line!
    "http://api:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- END CORS CONFIGURATION ---


# --- Pydantic Schema for Requests and Responses ---

class RecommendationRequest(BaseModel):
    """Schema for the input request."""
    # Note: In a real-world scenario, the userId would come from an authenticated session.
    # For this project, we accept a simple userId.
    user_id: int

class Recommendation(BaseModel):
    """Schema for a single recommendation."""
    movie_id: int
    title: str
    predicted_rating: float

class RecommendationResponse(BaseModel):
    """Schema for the recommendation list."""
    user_id: int
    recommendations: List[Recommendation]


# --- Lifecycle Hook: Model Loading ---

@app.on_event("startup")
async def load_models():
    """
    Loads the trained SVD model, Faiss index, and movie metadata into memory.
    This runs only once when the FastAPI application starts.
    """
    global svd_model, faiss_index, movie_map
    print("--- FrameRate: Loading Models for Inference ---")
    
    # 1. Check for artifacts existence
    if not SVD_MODEL_PATH.exists() or not FAISS_INDEX_PATH.exists() or not MOVIE_MAP_PATH.exists():
        print(f"CRITICAL ERROR: One or more model artifacts not found in '{MODEL_DIR}'.")
        print("Please ensure you have run 'python src/recommender/pipeline_train.py' successfully.")
        return

    # 2. Load SVD Model
    try:
        with open(SVD_MODEL_PATH, "rb") as f:
            svd_model = pickle.load(f)
        print(f"Successfully loaded SVD Model from {SVD_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading SVD model: {e}")
        return

    # 3. Load Faiss Index
    try:
        # We need the dimensionality of the SVD model's item vectors (qi)
        d = svd_model.qi.shape[1]
        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        
        # Simple check to confirm the index loaded correctly
        if faiss_index.ntotal != svd_model.qi.shape[0]:
            print("WARNING: Faiss index count does not match SVD item vector count.")
            
        print(f"Successfully loaded Faiss Index with {faiss_index.ntotal} vectors.")
    except Exception as e:
        print(f"Error loading Faiss index: {e}")
        return
        
    # 4. Load Movie Map (movieId to title)
    try:
        movie_map = pd.read_pickle(MOVIE_MAP_PATH)
        print(f"Successfully loaded Movie Map with {len(movie_map)} entries.")
    except Exception as e:
        print(f"Error loading Movie Map: {e}")
        return

    print("--- All Models and Artifacts Loaded Successfully. API Ready. ---")


# --- Core Prediction Utility ---

def get_user_vector(user_id: int) -> np.ndarray:
    """
    Retrieves the user vector (pu) from the SVD model, handling caching and
    conversion from raw userId to Surprise's internal ID.
    """
    global user_vector_cache, svd_model
    
    if user_id in user_vector_cache:
        return user_vector_cache[user_id]
        
    try:
        # Surprise's internal ID for the user
        inner_uid = svd_model.trainset.to_inner_uid(user_id)
        
        # SVD's pu matrix holds the user vectors
        user_vec = svd_model.pu[inner_uid]
        
        # Cache the result and return
        user_vector_cache[user_id] = user_vec
        return user_vec
        
    except ValueError:
        # This occurs if the user_id was not in the training data
        raise HTTPException(
            status_code=404, 
            detail=f"User ID {user_id} not found in training dataset. Cannot generate recommendations."
        )


# --- API Endpoint ---

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_films(request: RecommendationRequest):
    """
    Generates the top N film recommendations for a given user ID.
    """
    if not svd_model or not faiss_index or movie_map is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not yet loaded or failed to load on startup. Service unavailable."
        )
        
    user_id = request.user_id
    
    # 1. Get the User Vector
    # This represents the user's taste profile
    user_vec = get_user_vector(user_id)
    
    # Faiss requires the query vector to be reshaped for the search function
    query_vec = user_vec.reshape(1, -1)
    
    # 2. Faiss Search (The Sub-50ms Step)
    # D: Distances (or "scores"), I: Indices (the inner IDs of the nearest items)
    # We negate the vector before searching since SVD embeddings work with
    # cosine similarity, which is equivalent to L2 distance on normalized vectors.
    # For simplicity, we assume L2 distance here, which is standard for Faiss.
    # The Faiss index finds the N_RECOMMENDATIONS nearest neighbors.
    distances, inner_movie_ids = faiss_index.search(query_vec, N_RECOMMENDATIONS)
    
    # The results are wrapped in a list because faiss_index.search can take multiple queries
    inner_movie_ids = inner_movie_ids[0]
    distances = distances[0]
    
    # 3. Process and Reconstruct Recommendations
    recommendations = []
    
    # Map the inner IDs back to raw MovieLens IDs and titles
    for i, inner_id in enumerate(inner_movie_ids):
        # a. Get the raw (external) movieId from Surprise's mapping
        raw_movie_id = svd_model.trainset.to_raw_iid(inner_id)
        
        # b. Get the predicted rating (for presentation/sorting)
        # We use the SVD model's prediction function. The score from Faiss (distance) 
        # is related, but the SVD function gives the final predicted rating (1-5 scale).
        predicted_rating = svd_model.predict(user_id, raw_movie_id).est
        
        # c. Get the movie title
        try:
            movie_title = movie_map.loc[raw_movie_id]['title']
        except KeyError:
            movie_title = "Title Not Found" # Fallback
            
        recommendations.append(
            Recommendation(
                movie_id=raw_movie_id,
                title=movie_title,
                predicted_rating=round(predicted_rating, 2)
            )
        )
        
    # Optional: Sort by predicted rating (descending) if Faiss distance wasn't perfect
    recommendations.sort(key=lambda x: x.predicted_rating, reverse=True)
    
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations
    )
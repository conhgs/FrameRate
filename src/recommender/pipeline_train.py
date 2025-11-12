import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from surprise import SVD
from surprise.model_selection import cross_validate

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import log_model
from mlflow import pyfunc

from .utils import load_data_for_training, get_inner_id_map

# --- 1. CONFIGURATION ---
# The number of latent factors (k) is a key hyperparameter in SVD.
# It determines the dimensionality of your user and movie vectors.
N_FACTORS = 50 
N_EPOCHS = 20
LEARNING_RATE = 0.005
REGULARIZATION = 0.02
MLFLOW_EXPERIMENT_NAME = "FrameRate_Recommender_Training"

# Paths to save the final artifacts
MODEL_DIR = Path("models")
SVD_MODEL_PATH = MODEL_DIR / "svd_model.pkl"
FAISS_INDEX_PATH = MODEL_DIR / "faiss_index.bin"
MOVIE_MAP_PATH = MODEL_DIR / "movie_map.pkl"


# --- 2. FAISS UTILITY CLASS ---

class FaissSVDModel(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow model wrapper to save/load both the SVD model, 
    the Faiss index, and necessary metadata. This ensures everything 
    needed for prediction is logged together.
    """
    def __init__(self, svd_model, faiss_index, movie_id_map):
        self.svd_model = svd_model
        self.faiss_index = faiss_index
        self.movie_id_map = movie_id_map

    def predict(self, context, model_input):
        """
        Placeholder for the prediction logic. In production, this would 
        take a userId and return a list of recommended movieIds.
        """
        # This function is primarily for MLflow's internal signature logging
        # and testing. The actual API will use the artifacts directly.
        pass


def build_faiss_index(svd_model, trainset):
    """
    Extracts item vectors from the SVD model and builds a Faiss Index.
    
    Args:
        svd_model (surprise.prediction_algorithms.matrix_factorization.SVD): 
            The trained SVD model.
        trainset (surprise.Trainset): The full training set used to map IDs.
        
    Returns:
        faiss.IndexFlatL2: The built Faiss index for fast search.
    """
    print("\n--- Building Faiss Index ---")
    
    # 1. Get the item vectors
    # SVD"s qi matrix holds the item (movie) vectors.
    item_vectors = svd_model.qi
    
    # 2. Define dimensionality (should match N_FACTORS)
    d = item_vectors.shape[1]
    print(f"Item vector dimensionality (d): {d}")
    
    # 3. Initialize Faiss Index
    # IndexFlatL2 uses the L2 (Euclidean) distance, suitable for SVD embeddings.
    index = faiss.IndexFlatL2(d)
    
    # 4. Add vectors to the index
    index.add(item_vectors)
    print(f"Faiss index built and contains {index.ntotal} vectors.")
    
    return index


# --- 3. MAIN PIPELINE FUNCTION ---

def run_training_pipeline():
    """
    Main function to orchestrate data loading, SVD training, Faiss indexing, 
    and artifact logging via MLflow.
    """
    # 1. Setup MLflow Tracking
    # Ensure all artifacts and runs are logged under a clear experiment name
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Log Hyperparameters
        mlflow.log_param("n_factors", N_FACTORS)
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("lr_all", LEARNING_RATE)
        mlflow.log_param("reg_all", REGULARIZATION)

        # 2. Data Loading and Preparation
        try:
            trainset, full_data, movie_map_df = load_data_for_training()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return # Exit if data is missing

        # Save the movie map for API lookups (maps movieId to title)
        movie_map_df.to_pickle(MOVIE_MAP_PATH)
        mlflow.log_artifact(str(MOVIE_MAP_PATH), "movie_map")
        print(f"Saved movie map to {MOVIE_MAP_PATH}")

        # 3. SVD Model Training
        print("\n--- Training SVD Model ---")
        svd_model = SVD(
            n_factors=N_FACTORS, 
            n_epochs=N_EPOCHS, 
            lr_all=LEARNING_RATE, 
            reg_all=REGULARIZATION, 
            random_state=42
        )
        
        # Train on the full dataset
        svd_model.fit(trainset)
        
        # 4. Model Evaluation (using cross-validation for robustness)
        print("\n--- Evaluating Model (Cross Validation) ---")
        cv_results = cross_validate(
            svd_model, full_data, measures=["RMSE", "MAE"], cv=3, verbose=True, n_jobs=-1
        )

        # Log Metrics
        rmse = cv_results["test_rmse"].mean()
        mae = cv_results["test_mae"].mean()
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        print(f"Mean RMSE (3-fold CV): {rmse:.4f}")
        print(f"Mean MAE (3-fold CV): {mae:.4f}")

        # 5. Build and Save Faiss Index
        faiss_index = build_faiss_index(svd_model, trainset)
        faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
        
        # Log Faiss index as an artifact
        mlflow.log_artifact(str(FAISS_INDEX_PATH), "faiss_index")
        print(f"Saved Faiss index to {FAISS_INDEX_PATH}")
        
        # 6. Save and Log SVD Model
        # The SVD model needs to be pickled to be saved and logged
        with open(SVD_MODEL_PATH, "wb") as f:
            pickle.dump(svd_model, f)
        
        mlflow.log_artifact(str(SVD_MODEL_PATH), "svd_model")
        print(f"Saved SVD model to {SVD_MODEL_PATH}")

        # Final Log: Wrap the model for MLflow model tracking (optional but good practice)
        # We"ll log the components manually above, but we log the wrapper for completeness
        # log_model(
        #     python_model=FaissSVDModel(svd_model, faiss_index, movie_map_df), 
        #     artifact_path="recommender_model",
        # )

if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    
    run_training_pipeline()
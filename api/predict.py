"""
FastAPI backend for BERT vs DeBERTa fake review detection comparison
This can be deployed separately (Railway, Render, etc.)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
from transformers import DebertaModel, DebertaTokenizerFast, AutoModel, BertTokenizerFast
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List
import subprocess
import urllib.request

# Get the project root directory
# In backend folder structure: api/predict.py -> backend/models
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Global model variables (BERT and DeBERTa)
bert_model = None
bert_tokenizer = None
deberta_model = None
deberta_tokenizer = None
bert_loaded = False
deberta_loaded = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Memory optimization: Force CPU on Render (free tier has no GPU)
# Set FORCE_CPU=true in Render environment variables to use CPU
FORCE_CPU = os.environ.get("FORCE_CPU", "true").lower() == "true"
if FORCE_CPU:
    device = torch.device('cpu')
    print(f"[INFO] Using CPU device (FORCE_CPU=true)")

# Load only one model at a time to save memory
# Set LOAD_MODELS=bert or LOAD_MODELS=deberta or LOAD_MODELS=both
LOAD_MODELS = os.environ.get("LOAD_MODELS", "both").lower()

# Google Drive File IDs for automatic model download (set in Render environment variables)
DEBERTA_MODEL_GDRIVE_ID = os.environ.get("DEBERTA_MODEL_GDRIVE_ID", "")
BERT_MODEL_GDRIVE_ID = os.environ.get("BERT_MODEL_GDRIVE_ID", "")

# HuggingFace URLs for model download (fallback if Google Drive fails)
DEBERTA_MODEL_HF_URL = os.environ.get(
    "DEBERTA_MODEL_HF_URL", 
    "https://huggingface.co/imperfect0007/deberta/resolve/main/best_deberta_model.pt"
)
BERT_MODEL_HF_URL = os.environ.get(
    "BERT_MODEL_HF_URL",
    "https://huggingface.co/imperfect0007/bert/resolve/main/best_bert_model_lower_accuracy.pt"
)

def download_from_google_drive(file_id: str, output_path: Path):
    """Download file from Google Drive using File ID"""
    try:
        print(f"[INFO] Downloading model from Google Drive: {file_id}")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Create models directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download using urllib with timeout
        import socket
        socket.setdefaulttimeout(30)  # 30 second timeout
        urllib.request.urlretrieve(url, str(output_path))
        socket.setdefaulttimeout(None)  # Reset timeout
        
        # Check if file was downloaded (Google Drive may return HTML for large files)
        file_size = output_path.stat().st_size
        if file_size < 1024 * 1024:  # Less than 1MB, probably HTML error page
            print(f"[WARNING] Downloaded file is too small ({file_size} bytes). Trying alternative method...")
            
            # Try alternative method using wget or curl via subprocess
            try:
                # Use curl if available
                result = subprocess.run(
                    ['curl', '-L', '-c', '/tmp/cookies.txt', '-o', str(output_path),
                     f'https://drive.google.com/uc?export=download&id={file_id}'],
                    capture_output=True,
                    timeout=600  # 10 minute timeout for large files
                )
                if result.returncode == 0:
                    file_size = output_path.stat().st_size
                    if file_size > 1024 * 1024:  # Greater than 1MB
                        print(f"[SUCCESS] Downloaded model: {file_size / (1024*1024):.2f} MB")
                        return True
            except Exception as e:
                print(f"[ERROR] Alternative download method failed: {e}")
            
            # Delete the small file
            output_path.unlink()
            return False
        
        print(f"[SUCCESS] Downloaded model: {file_size / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download from Google Drive: {e}")
        return False

def download_from_huggingface(url: str, output_path: Path):
    """Download file from HuggingFace URL"""
    try:
        print(f"[INFO] Downloading model from HuggingFace: {url}")
        
        # Create models directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download using urllib with timeout
        import socket
        socket.setdefaulttimeout(120)  # 2 minute timeout for large files
        
        def reporthook(blocknum, blocksize, totalsize):
            """Progress hook for download"""
            if totalsize > 0:
                percent = min(100, (blocknum * blocksize * 100) / totalsize)
                if blocknum % 100 == 0:  # Print every 100 blocks
                    print(f"[INFO] Download progress: {percent:.1f}%")
        
        urllib.request.urlretrieve(url, str(output_path), reporthook=reporthook)
        socket.setdefaulttimeout(None)  # Reset timeout
        
        # Verify download was successful
        file_size = output_path.stat().st_size
        if file_size < 1024:  # Less than 1KB, probably an error page
            print(f"[ERROR] Downloaded file is too small ({file_size} bytes). Download may have failed.")
            output_path.unlink()
            return False
        
        print(f"[SUCCESS] Downloaded model from HuggingFace: {file_size / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download from HuggingFace: {e}")
        if output_path.exists():
            output_path.unlink()  # Clean up partial download
        return False

def ensure_models_downloaded():
    """Check if models exist, download from Google Drive if missing"""
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check DeBERTa model
        deberta_paths = [
            MODELS_DIR / "best_deberta_model.pt",
            MODELS_DIR / "complete_deberta_model.pth",
        ]
        deberta_exists = any(p.exists() for p in deberta_paths)
        
        if not deberta_exists and DEBERTA_MODEL_GDRIVE_ID:
            print("[INFO] DeBERTa model not found. Will download on first request if needed.")
            # Don't download during startup to avoid blocking
            # Models will be downloaded lazily when needed
        elif deberta_exists:
            print("[INFO] DeBERTa model file found")
        
        # Check BERT model
        bert_paths = [
            MODELS_DIR / "best_bert_model_lower_accuracy.pt",
            MODELS_DIR / "complete_bert_model.pth",
            MODELS_DIR / "complete_bert_model.pt",
        ]
        bert_exists = any(p.exists() for p in bert_paths)
        
        if not bert_exists and BERT_MODEL_GDRIVE_ID:
            print("[INFO] BERT model not found. Will download on first request if needed.")
            # Don't download during startup to avoid blocking
        elif bert_exists:
            print("[INFO] BERT model file found")
    except Exception as e:
        print(f"[WARNING] Error checking models: {e}")
        # Don't raise - allow server to start

# BERT Architecture (from your notebook)
class BERT_Arch(nn.Module):
    """
    Simplified BERT architecture for more realistic performance
    - Uses pre-trained BERT as feature extractor
    - Minimal classifier head to reduce capacity
    - Higher dropout for more regularization
    """
    
    def __init__(self, bert, dropout_rate=0.6, freeze_bert_layers=11):
        super(BERT_Arch, self).__init__()
        
        self.bert = bert
        
        # Freeze almost all BERT layers
        self._freeze_bert_layers(freeze_bert_layers)
        
        # High dropout for strong regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Very simple classifier (only one layer)
        self.fc = nn.Linear(768, 2)  # Direct from BERT to output
        
        # LogSoftmax for output probabilities
        self.softmax = nn.LogSoftmax(dim=1)
    
    def _freeze_bert_layers(self, num_layers_to_freeze):
        """Freeze the first num_layers_to_freeze BERT layers"""
        for i in range(num_layers_to_freeze):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
    
    def forward(self, sent_id, mask):
        """Forward pass with minimal processing"""
        # Pass through BERT
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        
        # Simple classification with high dropout
        x = self.dropout(cls_hs)  # 60% dropout
        x = self.fc(x)
        x = self.softmax(x)
        
        return x

# DeBERTa Architecture (from your notebook)
class DeBERTa_Arch(nn.Module):
    def __init__(self, deberta_model, dropout_rate=0.2):
        super(DeBERTa_Arch, self).__init__()
        self.deberta = deberta_model
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.8)
        self.dropout3 = nn.Dropout(dropout_rate * 0.6)
        
        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)
        
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        outputs = self.deberta(sent_id, attention_mask=mask)
        cls_hs = outputs.last_hidden_state[:, 0]
        
        x = self.bn1(cls_hs)
        x = self.dropout1(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc5(x)
        x = self.softmax(x)
        
        return x

def load_bert_model():
    """Load BERT model (lazy loading)"""
    global bert_model, bert_tokenizer, bert_loaded
    
    if bert_loaded:
        return
    
    try:
        print("Loading BERT model...")
        # Memory optimization: Clear cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load with low_cpu_mem_usage to save memory
        bert_base = AutoModel.from_pretrained('bert-base-uncased', low_cpu_mem_usage=True)
        bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        # Try different possible BERT model filenames
        bert_model_paths = [
            MODELS_DIR / "complete_bert_model.pth",
            MODELS_DIR / "best_bert_model_lower_accuracy.pt",
            MODELS_DIR / "complete_bert_model.pt",
        ]
        bert_model_path = None
        for path in bert_model_paths:
            if path.exists():
                bert_model_path = path
                break
        
        if not bert_model_path:
            # Try to download from Google Drive if File ID is configured
            output_path = MODELS_DIR / "best_bert_model_lower_accuracy.pt"
            download_success = False
            
            if BERT_MODEL_GDRIVE_ID:
                print("[INFO] BERT model not found. Attempting to download from Google Drive...")
                if download_from_google_drive(BERT_MODEL_GDRIVE_ID, output_path):
                    bert_model_path = output_path
                    download_success = True
            
            # If Google Drive download failed or not configured, try HuggingFace
            if not download_success and BERT_MODEL_HF_URL:
                print("[INFO] BERT model not found. Attempting to download from HuggingFace...")
                if download_from_huggingface(BERT_MODEL_HF_URL, output_path):
                    bert_model_path = output_path
                    download_success = True
            
            if not download_success:
                raise FileNotFoundError(
                    f"BERT model not found and all download methods failed. "
                    f"Tried: {[str(p) for p in bert_model_paths]}. "
                    f"Configure BERT_MODEL_GDRIVE_ID or BERT_MODEL_HF_URL environment variables."
                )
        print(f"Loading BERT model from: {bert_model_path}")
        # Use weights_only=False for PyTorch 2.6+ compatibility
        bert_checkpoint = torch.load(str(bert_model_path), map_location=device, weights_only=False)
        
        # Handle both .pth (with dict) and .pt (state_dict only) formats
        bert_model = BERT_Arch(bert_base, dropout_rate=0.6, freeze_bert_layers=11)
        
        if isinstance(bert_checkpoint, dict):
            # Check for different possible keys
            if 'model_state_dict' in bert_checkpoint:
                bert_model.load_state_dict(bert_checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in bert_checkpoint:
                bert_model.load_state_dict(bert_checkpoint['state_dict'], strict=False)
            else:
                # Try loading the entire dict as state_dict
                try:
                    bert_model.load_state_dict(bert_checkpoint, strict=False)
                except Exception as e:
                    print(f"Warning: Could not load BERT from dict format: {e}")
                    print("Available keys in checkpoint:", list(bert_checkpoint.keys())[:10])
                    raise
        else:
            # It's a direct state_dict
            bert_model.load_state_dict(bert_checkpoint, strict=False)
        
        bert_model.to(device)
        bert_model.eval()
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bert_loaded = True
        print("[SUCCESS] BERT model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        raise

def load_deberta_model():
    """Load DeBERTa model (lazy loading)"""
    global deberta_model, deberta_tokenizer, deberta_loaded
    
    if deberta_loaded:
        return
    
    try:
        print("Loading DeBERTa model...")
        # Memory optimization: Clear cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load with low_cpu_mem_usage to save memory
        deberta_base = DebertaModel.from_pretrained('microsoft/deberta-base', low_cpu_mem_usage=True)
        deberta_tokenizer = DebertaTokenizerFast.from_pretrained('microsoft/deberta-base')
        
        # Try different possible DeBERTa model filenames
        deberta_model_paths = [
            MODELS_DIR / "complete_deberta_model.pth",
            MODELS_DIR / "best_deberta_model.pt",
            MODELS_DIR / "complete_deberta_model.pt",
        ]
        deberta_model_path = None
        for path in deberta_model_paths:
            if path.exists():
                deberta_model_path = path
                break
        
        if not deberta_model_path:
            # Try to download from Google Drive if File ID is configured
            output_path = MODELS_DIR / "best_deberta_model.pt"
            download_success = False
            
            if DEBERTA_MODEL_GDRIVE_ID:
                print("[INFO] DeBERTa model not found. Attempting to download from Google Drive...")
                if download_from_google_drive(DEBERTA_MODEL_GDRIVE_ID, output_path):
                    deberta_model_path = output_path
                    download_success = True
            
            # If Google Drive download failed or not configured, try HuggingFace
            if not download_success and DEBERTA_MODEL_HF_URL:
                print("[INFO] DeBERTa model not found. Attempting to download from HuggingFace...")
                if download_from_huggingface(DEBERTA_MODEL_HF_URL, output_path):
                    deberta_model_path = output_path
                    download_success = True
            
            if not download_success:
                raise FileNotFoundError(
                    f"DeBERTa model not found and all download methods failed. "
                    f"Tried: {[str(p) for p in deberta_model_paths]}. "
                    f"Configure DEBERTA_MODEL_GDRIVE_ID or DEBERTA_MODEL_HF_URL environment variables."
                )
        print(f"Loading DeBERTa model from: {deberta_model_path}")
        # Use weights_only=False for PyTorch 2.6+ compatibility
        deberta_checkpoint = torch.load(str(deberta_model_path), map_location=device, weights_only=False)
        
        # Handle both .pth (with dict) and .pt (state_dict only) formats
        deberta_model = DeBERTa_Arch(deberta_base, dropout_rate=0.2)
        
        if isinstance(deberta_checkpoint, dict):
            # Check for different possible keys
            if 'model_state_dict' in deberta_checkpoint:
                deberta_model.load_state_dict(deberta_checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in deberta_checkpoint:
                deberta_model.load_state_dict(deberta_checkpoint['state_dict'], strict=False)
            else:
                # Try loading the entire dict as state_dict
                try:
                    deberta_model.load_state_dict(deberta_checkpoint, strict=False)
                except Exception as e:
                    print(f"Warning: Could not load from dict format: {e}")
                    print("Available keys in checkpoint:", list(deberta_checkpoint.keys())[:10])
                    raise
        else:
            # It's a direct state_dict
            deberta_model.load_state_dict(deberta_checkpoint, strict=False)
        
        deberta_model.to(device)
        deberta_model.eval()
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        deberta_loaded = True
        print("[SUCCESS] DeBERTa model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading DeBERTa model: {e}")
        raise

def load_models():
    """Load models based on LOAD_MODELS environment variable"""
    global bert_loaded, deberta_loaded
    
    if LOAD_MODELS in ["bert", "both"]:
        try:
            load_bert_model()
        except Exception as e:
            print(f"Warning: Could not load BERT model: {e}")
    
    if LOAD_MODELS in ["deberta", "both"]:
        try:
            load_deberta_model()
        except Exception as e:
            print(f"Warning: Could not load DeBERTa model: {e}")

# Load models on startup (optional - can be disabled for memory-constrained environments)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Download models from Google Drive if not present (non-blocking)
    print("[INFO] Checking for model files...")
    try:
        # Run model download check in a way that doesn't block server startup
        ensure_models_downloaded()
    except Exception as e:
        print(f"[WARNING] Model download check failed (non-fatal): {e}")
        print("[INFO] Server will continue. Models will be downloaded on first request if needed.")
    
    # Startup: Optionally load models
    # Set LOAD_ON_STARTUP=false to disable loading on startup (lazy load instead)
    load_on_startup = os.environ.get("LOAD_ON_STARTUP", "false").lower() == "true"
    
    if load_on_startup:
        try:
            print("[INFO] Loading models on startup...")
            load_models()
        except Exception as e:
            print(f"[WARNING] Failed to load models on startup: {e}")
            print("[INFO] Models will be loaded lazily on first request")
    else:
        print("[INFO] Skipping model loading on startup (lazy loading enabled)")
        print("[INFO] Models will be loaded on first prediction request")
    
    print("[INFO] FastAPI server is ready to accept requests")
    yield
    # Shutdown: cleanup if needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="BERT vs DeBERTa Fake Review Detection API", lifespan=lifespan)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str

class ModelPrediction(BaseModel):
    prediction: int
    confidence: float
    probabilities: List[float]

class PredictionResponse(BaseModel):
    bert: ModelPrediction
    deberta: ModelPrediction

def predict_text(text: str, model, tokenizer, max_length: int = 244):
    """Make prediction on a single text"""
    try:
        model.eval()
        
        # Tokenize
        inputs = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probabilities = torch.exp(outputs).cpu().numpy()[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": [float(probabilities[0]), float(probabilities[1])]
        }
    except Exception as e:
        print(f"Error in predict_text: {e}")
        print(f"Input text: {text[:100]}...")
        raise

@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    models_status = []
    if bert_loaded or LOAD_MODELS in ["bert", "both"]:
        models_status.append("BERT")
    if deberta_loaded or LOAD_MODELS in ["deberta", "both"]:
        models_status.append("DeBERTa")
    
    response = {
        "message": "BERT vs DeBERTa Fake Review Detection API", 
        "status": "running",
        "models": models_status,
        "bert_loaded": bert_loaded,
        "deberta_loaded": deberta_loaded,
        "device": str(device)
    }
    return response

@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    """Health check endpoint with model status"""
    response = {
        "status": "healthy",
        "bert_loaded": bert_loaded,
        "deberta_loaded": deberta_loaded,
        "device": str(device),
        "load_models": LOAD_MODELS
    }
    return response

@app.get("/favicon.ico")
def favicon():
    """Handle favicon requests to avoid 404 errors"""
    from fastapi.responses import Response
    return Response(status_code=204)  # No Content

@app.get("/dataset")
async def get_dataset():
    """Get dataset samples and statistics"""
    try:
        # Dataset is in backend folder
        dataset_path = BASE_DIR / "newdataset.csv"
        # Also try root directory if moved
        if not dataset_path.exists():
            dataset_path = BASE_DIR.parent / "newdataset.csv"
        
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        # Load first 100 samples for display
        df_samples = pd.read_csv(dataset_path, nrows=100)
        
        # Get full statistics efficiently
        df_full = pd.read_csv(dataset_path)
        total_genuine = int((df_full['label'] == 1).sum())
        total_fake = int((df_full['label'] == 0).sum())
        
        # Format samples
        samples = []
        for _, row in df_samples.iterrows():
            text = str(row['text']) if pd.notna(row['text']) else ''
            samples.append({
                "category": str(row['category']) if pd.notna(row['category']) else 'Unknown',
                "rating": float(row['rating']) if pd.notna(row['rating']) else 0.0,
                "text": text[:500],  # Truncate long texts
                "label": int(row['label']) if pd.notna(row['label']) else 0
            })
        
        return {
            "stats": {
                "total": len(df_full),
                "genuine": total_genuine,
                "fake": total_fake,
                "categories": int(df_full['category'].nunique())
            },
            "samples": samples
        }
    except Exception as e:
        print(f"Dataset error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """BERT vs DeBERTa comparison prediction for fake review detection"""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Lazy load models if not already loaded
    if LOAD_MODELS in ["bert", "both"]:
        if not bert_loaded:
            try:
                load_bert_model()
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Failed to load BERT model: {str(e)}")
    
    if LOAD_MODELS in ["deberta", "both"]:
        if not deberta_loaded:
            try:
                load_deberta_model()
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Failed to load DeBERTa model: {str(e)}")
    
    # Check if models are loaded
    if LOAD_MODELS in ["bert", "both"]:
        if bert_model is None or bert_tokenizer is None:
            raise HTTPException(status_code=503, detail="BERT model not loaded yet. Please wait...")
    
    if LOAD_MODELS in ["deberta", "both"]:
        if deberta_model is None or deberta_tokenizer is None:
            raise HTTPException(status_code=503, detail="DeBERTa model not loaded yet. Please wait...")
    
    try:
        results = {}
        
        # Get BERT prediction (if enabled)
        if LOAD_MODELS in ["bert", "both"]:
            bert_result = predict_text(
                request.text,
                bert_model,
                bert_tokenizer,
                max_length=244
            )
            results["bert"] = bert_result
        
        # Get DeBERTa prediction (if enabled)
        if LOAD_MODELS in ["deberta", "both"]:
            deberta_result = predict_text(
                request.text,
                deberta_model,
                deberta_tokenizer,
                max_length=244
            )
            results["deberta"] = deberta_result
        
        # Return only loaded models
        if len(results) == 1:
            # If only one model, return it directly
            return results
        else:
            return results
    except Exception as e:
        print(f"Prediction error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Render sets PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)



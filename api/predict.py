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

# Get the project root directory
# In backend folder structure: api/predict.py -> backend/models
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Global model variables (BERT and DeBERTa)
bert_model = None
bert_tokenizer = None
deberta_model = None
deberta_tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def load_models():
    """Load both BERT and DeBERTa models"""
    global bert_model, bert_tokenizer, deberta_model, deberta_tokenizer
    
    try:
        # Load BERT
        print("Loading BERT model...")
        bert_base = AutoModel.from_pretrained('bert-base-uncased')
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
            raise FileNotFoundError(f"BERT model not found. Tried: {[str(p) for p in bert_model_paths]}")
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
        print("[SUCCESS] BERT model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        raise
    
    try:
        # Load DeBERTa
        print("Loading DeBERTa model...")
        deberta_base = DebertaModel.from_pretrained('microsoft/deberta-base')
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
            raise FileNotFoundError(f"DeBERTa model not found. Tried: {[str(p) for p in deberta_model_paths]}")
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
        print("[SUCCESS] DeBERTa model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading DeBERTa model: {e}")
        raise

# Load models on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    try:
        load_models()
    except Exception as e:
        print(f"Failed to load models: {e}")
        # Don't fail startup if models fail to load
    yield
    # Shutdown: cleanup if needed

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

@app.get("/")
def read_root():
    return {
        "message": "BERT vs DeBERTa Fake Review Detection API", 
        "status": "running", 
        "models": ["BERT", "DeBERTa"]
    }

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

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """BERT vs DeBERTa comparison prediction for fake review detection"""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    if bert_model is None or bert_tokenizer is None:
        raise HTTPException(status_code=503, detail="BERT model not loaded yet. Please wait...")
    
    if deberta_model is None or deberta_tokenizer is None:
        raise HTTPException(status_code=503, detail="DeBERTa model not loaded yet. Please wait...")
    
    try:
        # Get BERT prediction
        bert_result = predict_text(
            request.text,
            bert_model,
            bert_tokenizer,
            max_length=244
        )
        
        # Get DeBERTa prediction
        deberta_result = predict_text(
            request.text,
            deberta_model,
            deberta_tokenizer,
            max_length=244
        )
        
        return {
            "bert": bert_result,
            "deberta": deberta_result
        }
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



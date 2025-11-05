# Backend - DeBERTa Fake Review Detection API

FastAPI backend for fake review detection using DeBERTa model.

## Structure

```
backend/
├── api/
│   └── predict.py      # Main API application
├── models/             # Your .pt/.pth model files go here
│   ├── complete_deberta_model.pth
│   └── best_deberta_model.pt
├── newdataset.csv      # Dataset file
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── railway.json        # Railway deployment config
└── Procfile            # Process file for deployment
```

## Local Development

```bash
cd backend
pip install -r requirements.txt
cd api
python predict.py
```

Backend will run on http://localhost:8000

## Deployment

### Render

1. Connect GitHub repo
2. Render will detect `render.yaml`
3. Or manually set:
   - **Root Directory**: Leave empty
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd api && python predict.py`

### Railway

1. Connect GitHub repo
2. Set root directory to `backend/`
3. Start command: `cd api && python predict.py`

## Model Files

Upload your model files to `models/` directory:
- `complete_deberta_model.pth`
- `best_deberta_model.pt`

## Environment Variables

- `PORT`: Set automatically by hosting platform
- No other variables needed by default

## API Endpoints

- `GET /` - Health check
- `GET /dataset` - Get dataset samples and statistics
- `POST /predict` - Make prediction on text



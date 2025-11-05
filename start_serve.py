#!/usr/bin/env python3
"""
Startup script for Render deployment
Reads PORT from environment and starts uvicorn server
"""
import os
import sys
from pathlib import Path

# Change to api directory
api_dir = Path(__file__).parent / "api"
if api_dir.exists():
    os.chdir(api_dir)
    sys.path.insert(0, str(api_dir))

# Import and start
import uvicorn
from predict import app

port = int(os.environ.get("PORT", 8000))
print(f"[INFO] Starting server on port {port}")

uvicorn.run(app, host="0.0.0.0", port=port)


#!/usr/bin/env python
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

# Import and start with error handling
try:
    import uvicorn
    # Load FastAPI app from predict.py via file path to avoid import resolution issues
    import importlib.util
    predict_path = str(api_dir / "predict.py")
    spec = importlib.util.spec_from_file_location("predict", predict_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fastapi_app = getattr(module, "app")
    else:
        raise ImportError("Unable to load predict module from path: " + predict_path)
    
    port = int(os.environ.get("PORT", 8000))
    print(f"[INFO] Starting server on port {port}")
    print(f"[INFO] Working directory: {os.getcwd()}")
    print(f"[INFO] Python path: {sys.path[:3]}")
    
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port, log_level="info")
except ImportError as e:
    print(f"[ERROR] Failed to import dependencies: {e}")
    print(f"[ERROR] Python path: {sys.path}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Failed to start server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


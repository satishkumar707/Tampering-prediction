from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from lr_detector import LRDetector

app = FastAPI(title="Splicing Detection API", description="API for detecting image tampering using Logistic Regression")

# Initialize detector once at startup
detector = None

@app.on_event("startup")
async def load_model():
    global detector
    try:
        detector = LRDetector()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # In production this might be fatal, but here we can try-catch
        # If the model path is wrong, it will fail here.

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not detector:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Run analysis
        result = detector.analyze(tmp_path)
        
        # Clean up result for JSON response (convert numpy types/Infinity to standard types)
        # Note: detector already returns standard python types for scores, so just need to ensure cleanliness
        response = {
            "filename": file.filename,
            "verdict": result['verdict'],
            "confidence": result['confidence'],
            "tampering_probability": result['prob_tampered'],
            "details": {
                "dct_score": result['dct_score'],
                "ghost_score": result['ghost_score'],
                "ratio": result['ratio'],
                "latency_seconds": result['latency']
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
    finally:
        # Cleanup temp file
        Path(tmp_path).unlink(missing_ok=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

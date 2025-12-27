"""Inference script using Logistic Regression model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from splicing_detector import SplicingDetector


class LRDetector:
    """Detector using a trained Logistic Regression model."""
    
    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = str(Path(__file__).resolve().parent / 'tampering_model.joblib')
        print(f"Loading Logistic Regression model from {model_path}...")
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.detector = SplicingDetector()
        print("Model loaded successfully.")

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """Analyze image and predict using Logistic Regression."""
        start_time = time.time()
        
        # Extract features
        res = self.detector.analyze(image_path)
        dct = res['dct_score']
        ghost = res['ghost_score']
        pct = res['ghost_suspicious_pct']
        ratio = ghost / (dct + 1e-6)
        
        # Prepare feature vector
        feats = np.array([[dct, ghost, pct, ratio]])
        feats_scaled = self.scaler.transform(feats)
        
        # Prediction
        prob = self.model.predict_proba(feats_scaled)[0][1]
        verdict_idx = self.model.predict(feats_scaled)[0]
        
        verdict = "TAMPERED" if verdict_idx == 1 else "AUTHENTIC"
        
        # Determine confidence based on probability
        if prob > 0.8 or prob < 0.2:
            confidence = "HIGH"
        elif prob > 0.6 or prob < 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
            
        latency = time.time() - start_time
        
        return {
            'image_name': Path(image_path).name,
            'dct_score': dct,
            'ghost_score': ghost,
            'ratio': ratio,
            'prob_tampered': float(prob),
            'verdict': verdict,
            'confidence': confidence,
            'latency': latency
        }


def main():
    detector = LRDetector()
    # Resolve data path relative to this script
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / 'data/'
    
    # Ground Truth for verification
    ground_truth = {
        '1.jpeg': 'TAMPERED', '2.jpeg': 'TAMPERED', '3.jpeg': 'AUTHENTIC',
        '4.jpeg': 'TAMPERED', '5.jpeg': 'TAMPERED', '6.jpeg': 'TAMPERED',
        '7.jpeg': 'TAMPERED', '8.jpeg': 'AUTHENTIC','9.jpg': 'AUTHENTIC',
        '10.jpg': 'AUTHENTIC'
    }
    
    image_files = sorted([
        data_dir / f for f in ground_truth.keys() if (data_dir / f).exists()
    ])
    
    results = []
    print(f"\nRunning LR Inference on {len(image_files)} images...")
    
    for img_path in tqdm(image_files, desc="Processing"):
        results.append(detector.analyze(str(img_path)))
    
    # Save results
    df = pd.DataFrame(results)
    output_path = project_root / 'results' / 'lr_results.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Report
    print(f"\n{'='*80}")
    print(f"{'Image':<25} {'Verdict':<15} {'Prob':<10} {'Latency':<10} {'Match'}")
    print("-" * 80)
    
    correct = 0
    for r in results:
        name = r['image_name']
        pred = r['verdict']
        prob = r['prob_tampered']
        lat = r['latency']
        gt = ground_truth.get(name, 'UNKNOWN')
        match = "✓" if pred == gt else "✗"
        if pred == gt: correct += 1
        
        print(f"{name:<25} {pred:<15} {prob:<10.3f} {lat:<10.3f} {match}")
    
    print(f"\n{'='*80}")
    print(f"Final LR Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

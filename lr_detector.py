"""Inference script using Logistic Regression model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

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
        
        res = self.detector.analyze(image_path)
        dct = res['dct_score']
        ghost = res['ghost_score']
        pct = res['ghost_suspicious_pct']
        ratio = ghost / (dct + 1e-6)
        
        feats = np.array([[dct, ghost, pct, ratio]])
        feats_scaled = self.scaler.transform(feats)
        
        prob = self.model.predict_proba(feats_scaled)[0][1]
        verdict_idx = self.model.predict(feats_scaled)[0]
        
        verdict = "TAMPERED" if verdict_idx == 1 else "AUTHENTIC"
        
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


def load_ground_truth(gt_path: str) -> Dict[str, str]:
    """Load ground truth from CSV file."""
    df = pd.read_csv(gt_path)
    required_cols = {'file_name', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Ground truth CSV must contain columns: {required_cols}")
    return dict(zip(df['file_name'], df['label'].str.upper()))


def parse_args():
    parser = argparse.ArgumentParser(description="Run LR inference on images")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/test',
        help='Path to data folder containing images (default: data/test)'
    )
    parser.add_argument(
        '--gt-csv',
        type=str,
        default=None,
        help='Path to ground truth CSV with columns: file_name, label (optional)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model file (default: tampering_model.joblib)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    detector = LRDetector(model_path=args.model_path)
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    ground_truth: Optional[Dict[str, str]] = None
    if args.gt_csv:
        gt_path = Path(args.gt_csv)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        ground_truth = load_ground_truth(args.gt_csv)
        print(f"Loaded ground truth for {len(ground_truth)} images.")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = sorted([
        f for f in data_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        raise ValueError(f"No images found in {data_dir}")
    
    results = []
    print(f"\nRunning LR Inference on {len(image_files)} images from {data_dir}...")
    
    for img_path in tqdm(image_files, desc="Processing"):
        results.append(detector.analyze(str(img_path)))
    
    df = pd.DataFrame(results)
    
    if ground_truth:
        df['ground_truth'] = df['image_name'].map(ground_truth)
        df['correct'] = df['verdict'] == df['ground_truth']
    
    output_path = Path(args.output_dir) / 'lr_results.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    if ground_truth:
        print(f"\n{'='*85}")
        print(f"{'Image':<30} {'Verdict':<12} {'GT':<12} {'Prob':<10} {'Latency':<10} {'Match'}")
        print("-" * 85)
        
        correct = 0
        total_with_gt = 0
        for r in results:
            name = r['image_name']
            pred = r['verdict']
            prob = r['prob_tampered']
            lat = r['latency']
            gt = ground_truth.get(name)
            
            if gt:
                total_with_gt += 1
                match = "✓" if pred == gt else "✗"
                if pred == gt:
                    correct += 1
            else:
                gt = "N/A"
                match = "-"
            
            print(f"{name:<30} {pred:<12} {gt:<12} {prob:<10.3f} {lat:<10.3f} {match}")
        
        print(f"{'='*85}")
        if total_with_gt > 0:
            accuracy = correct / total_with_gt * 100
            print(f"Accuracy: {correct}/{total_with_gt} ({accuracy:.1f}%)")
        print(f"{'='*85}\n")
    else:
        print(f"\n{'='*70}")
        print(f"{'Image':<30} {'Verdict':<12} {'Prob':<10} {'Latency':<10}")
        print("-" * 70)
        
        for r in results:
            print(f"{r['image_name']:<30} {r['verdict']:<12} {r['prob_tampered']:<10.3f} {r['latency']:<10.3f}")
        
        print(f"{'='*70}\n")
    
    print(f"Results saved to: {output_path}\n")


if __name__ == '__main__':
    main()
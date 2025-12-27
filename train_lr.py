"""Training script for Logistic Regression tampering detection model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

from splicing_detector import SplicingDetector


def load_ground_truth(gt_path: str) -> dict[str, int]:
    """Load ground truth from CSV file."""
    df = pd.read_csv(gt_path)
    required_cols = {'file_name', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Ground truth CSV must contain columns: {required_cols}")
    
    label_map = {'TAMPERED': 1, 'AUTHENTIC': 0}
    df['label_int'] = df['label'].str.upper().map(label_map)
    
    if df['label_int'].isna().any():
        invalid = df[df['label_int'].isna()]['label'].unique()
        raise ValueError(f"Invalid labels found: {invalid}. Expected 'TAMPERED' or 'AUTHENTIC'.")
    
    return dict(zip(df['file_name'], df['label_int'].astype(int)))


def extract_features(
    detector: SplicingDetector,
    data_dir: Path,
    ground_truth: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract features from images."""
    features, labels, names = [], [], []
    
    for name, label in ground_truth.items():
        path = data_dir / name
        if not path.exists():
            print(f"Warning: Image not found: {path}")
            continue
            
        res = detector.analyze(str(path))
        features.append([
            res['dct_score'],
            res['ghost_score'],
            res['ghost_suspicious_pct'],
            res['ghost_score'] / (res['dct_score'] + 1e-6)
        ])
        labels.append(label)
        names.append(name)
    
    return np.array(features), np.array(labels), names


def parse_args():
    parser = argparse.ArgumentParser(description="Train LR tampering detection model")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/train',
        help='Path to training data folder (default: data/train)'
    )
    parser.add_argument(
        '--gt-csv',
        type=str,
        required=True,
        help='Path to ground truth CSV with columns: file_name, label'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='tampering_model.joblib',
        help='Path to save trained model (default: tampering_model.joblib)'
    )
    parser.add_argument(
        '--class-weight-authentic',
        type=float,
        default=2.5,
        help='Class weight for authentic images to reduce FPR (default: 2.5)'
    )
    parser.add_argument(
        '--regularization',
        type=float,
        default=100.0,
        help='Regularization parameter C (default: 100.0)'
    )
    return parser.parse_args()


def train_model(args) -> None:
    """Train the LR model."""
    detector = SplicingDetector()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    gt_path = Path(args.gt_csv)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    ground_truth = load_ground_truth(args.gt_csv)
    print(f"Loaded ground truth for {len(ground_truth)} images.")
    
    print("\n[1/3] Extracting features...")
    X, y, names = extract_features(detector, data_dir, ground_truth)
    
    if len(X) == 0:
        raise ValueError(f"No features extracted! Check that images exist in {data_dir}")
    
    n_authentic = sum(y == 0)
    n_tampered = sum(y == 1)
    print(f"Dataset: {len(X)} images ({n_authentic} authentic, {n_tampered} tampered)")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('over', RandomOverSampler(random_state=42)),
        ('model', LogisticRegression(
            max_iter=10000,
            random_state=42,
            class_weight={0: args.class_weight_authentic, 1: 1.0},
            C=args.regularization
        ))
    ])
    
    print("[2/3] Training model...")
    pipeline.fit(X, y)
    
    print("[3/3] Evaluating on training data...")
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]
    
    acc = (preds == y).mean() * 100
    fp = sum((preds == 1) & (y == 0))
    fn = sum((preds == 0) & (y == 1))
    fpr = (fp / n_authentic * 100) if n_authentic > 0 else 0
    fnr = (fn / n_tampered * 100) if n_tampered > 0 else 0
    
    print(f"\n{'='*60}")
    print("TRAINING REPORT")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.1f}%")
    print(f"False Positive Rate: {fpr:.1f}%")
    print(f"False Negative Rate: {fnr:.1f}%")
    print(f"Correct: {sum(preds == y)}/{len(y)} | FP: {fp} | FN: {fn}")
    
    print(f"\n{'Image':<25} {'GT':<10} {'Pred':<10} {'Prob':<10} {'Match'}")
    print("-" * 60)
    for name, true, pred, prob in zip(names, y, preds, probs):
        gt_str = "TAMPERED" if true == 1 else "AUTHENTIC"
        pred_str = "TAMPERED" if pred == 1 else "AUTHENTIC"
        match = "✓" if true == pred else "✗"
        print(f"{name:<25} {gt_str:<10} {pred_str:<10} {prob:<10.3f} {match}")
    print(f"{'='*60}\n")
    
    model_data = {
        'model': pipeline.named_steps['model'],
        'scaler': pipeline.named_steps['scaler'],
        'feature_names': ['dct', 'ghost', 'pct', 'ratio']
    }
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, output_path)
    print(f"Model saved to: {output_path}")


def main():
    args = parse_args()
    train_model(args)


if __name__ == '__main__':
    main()
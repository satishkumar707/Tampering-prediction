"""Deep Learning-based tampering detection using pre-trained models from Hugging Face."""

import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


class DeepLearningDetector:
    """Orchestrates inference using multiple pre-trained models."""
    
    def __init__(self):
        print("Loading deep learning models from Hugging Face Hub...")
        # Model 1: ViT-based Deepfake Detector
        # This model is specifically fine-tuned for Real vs Deepfake/Synthetic
        self.model1_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        self.pipe1 = pipeline("image-classification", model=self.model1_name)
        
        # Model 2: General Forgery/AI Detector
        self.model2_name = "dima806/deepfake_vs_real_image_detection"
        self.pipe2 = pipeline("image-classification", model=self.model2_name)
        
        print("Models loaded successfully.")

    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """Run inference using all loaded models on a single image."""
        start_time = time.time()
        
        # Open image
        img = Image.open(image_path).convert("RGB")
        
        # Model 1 Inference
        res1 = self.pipe1(img)
        # Format: [{'label': 'label_name', 'score': 0.99}, ...]
        top1 = res1[0]
        
        # Model 2 Inference
        res2 = self.pipe2(img)
        top2 = res2[0]
        
        latency = time.time() - start_time
        
        return {
            "image_name": image_path.name,
            "m1_label": top1["label"],
            "m1_score": top1["score"],
            "m2_label": top2["label"],
            "m2_score": top2["score"],
            "dl_latency": latency
        }

    def run_inference(self, data_dir: Path) -> List[Dict[str, Any]]:
        """Run batch inference on all images in the specified directory."""
        image_files = sorted([
            f for f in data_dir.iterdir() 
            if f.suffix.lower() in [".jpeg", ".jpg", ".png"]
        ])
        
        results = []
        print(f"\nRunning DL inference on {len(image_files)} images...")
        
        for img_path in tqdm(image_files, desc="Processing"):
            try:
                result = self.analyze_image(img_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                
        return results


def main():
    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    detector = DeepLearningDetector()
    results = detector.run_inference(data_dir)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    output_csv = results_dir / "dl_results.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"\nDL Inference Complete. Results saved to: {output_csv}")
    
    # Simple summary
    print("\n" + "="*70)
    print(f"{'Image':<30} {'Model 1 (ViT)':<20} {'Model 2 (ViT)':<20}")
    print("-" * 70)
    for r in results:
        m1_info = f"{r['m1_label']} ({r['m1_score']:.2f})"
        m2_info = f"{r['m2_label']} ({r['m2_score']:.2f})"
        print(f"{r['image_name']:<30} {m1_info:<20} {m2_info:<20}")
    print("="*70)


if __name__ == "__main__":
    main()

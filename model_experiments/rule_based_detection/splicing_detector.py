"""Splicing detection using JPEG ghost analysis and DCT coefficients."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.fftpack import dct
from numpy.typing import NDArray


class JPEGGhostDetector:
    """Detects splicing by analyzing JPEG compression inconsistencies."""
    
    def __init__(self, quality_range: list[int] | None = None) -> None:
        self.quality_range = quality_range or list(range(60, 101, 5))
    
    def compute_ghost_map(self, image_path: str) -> dict[str, Any]:
        """Compute JPEG ghost map by testing multiple quality levels."""
        original = Image.open(image_path).convert('RGB')
        original_np = np.array(original).astype(np.float32)
        
        quality_diffs: dict[int, NDArray[np.float32]] = {}
        
        for quality in self.quality_range:
            buffer = io.BytesIO()
            original.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            
            resaved = Image.open(buffer).convert('RGB')
            resaved_np = np.array(resaved).astype(np.float32)
            
            diff = np.abs(original_np - resaved_np)
            quality_diffs[quality] = np.mean(diff, axis=2)
        
        quality_array = np.array(list(quality_diffs.values()))
        min_quality_idx = np.argmin(quality_array, axis=0)
        min_quality_map = np.array([
            self.quality_range[i] for i in min_quality_idx.flatten()
        ]).reshape(min_quality_idx.shape)
        
        variance = np.var(min_quality_map)
        quality_std = np.std(min_quality_map)
        quality_mean = np.mean(min_quality_map)
        
        suspicious_mask = np.abs(min_quality_map - quality_mean) > 1.5 * quality_std
        tampering_score = min(1.0, variance / 400.0)
        
        return {
            'min_quality_map': min_quality_map,
            'quality_variance': float(variance),
            'quality_mean': float(quality_mean),
            'quality_std': float(quality_std),
            'suspicious_mask': suspicious_mask,
            'suspicious_percentage': float(np.sum(suspicious_mask) / suspicious_mask.size * 100),
            'tampering_score': tampering_score
        }
    
    def analyze(self, image_path: str) -> dict[str, Any]:
        """Perform full JPEG ghost analysis."""
        result = self.compute_ghost_map(image_path)
        
        suspicious_mask = result['suspicious_mask'].astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(suspicious_mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_regions = [c for c in contours if cv2.contourArea(c) > 500]
        
        result['num_suspicious_regions'] = len(significant_regions)
        result['has_large_suspicious_region'] = any(
            cv2.contourArea(c) > 5000 for c in significant_regions
        )
        
        return result


class DCTAnalyzer:
    """Analyzes DCT coefficients for double JPEG compression artifacts."""
    
    def __init__(self, block_size: int = 8) -> None:
        self.block_size = block_size
    
    def extract_dct_coefficients(self, image: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Extract DCT coefficient variances from image blocks."""
        h, w = image.shape
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        
        dct_variances = np.zeros((blocks_h, blocks_w))
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = image[
                    i*self.block_size:(i+1)*self.block_size,
                    j*self.block_size:(j+1)*self.block_size
                ]
                
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_variances[i, j] = np.var(dct_block)
        
        return dct_variances
    
    def analyze(self, image_path: str) -> dict[str, Any]:
        """Analyze DCT coefficient patterns for double compression."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        dct_vars = self.extract_dct_coefficients(img)
        
        var_mean = np.mean(dct_vars)
        var_std = np.std(dct_vars)
        
        inconsistency_score = min(1.0, var_std / (var_mean + 1e-6))
        threshold = var_mean + 2 * var_std
        suspicious_blocks = dct_vars > threshold
        
        tampering_score = min(1.0, inconsistency_score + (np.sum(suspicious_blocks) / suspicious_blocks.size))
        
        return {
            'dct_mean_variance': float(var_mean),
            'dct_std_variance': float(var_std),
            'inconsistency_score': inconsistency_score,
            'suspicious_block_percentage': float(np.sum(suspicious_blocks) / suspicious_blocks.size * 100),
            'tampering_score': tampering_score
        }


class EdgeIrregularityDetector:
    """Detects irregular edges that may indicate splicing boundaries."""
    
    def analyze(self, image_path: str) -> dict[str, Any]:
        """Analyze edge patterns for splicing artifacts."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        h, w = edges.shape
        region_size = 64
        regions_h = h // region_size
        regions_w = w // region_size
        
        edge_densities = np.zeros((regions_h, regions_w))
        
        for i in range(regions_h):
            for j in range(regions_w):
                region = edges[
                    i*region_size:(i+1)*region_size,
                    j*region_size:(j+1)*region_size
                ]
                edge_densities[i, j] = np.sum(region) / (region_size * region_size * 255)
        
        density_variance = np.var(edge_densities)
        density_mean = np.mean(edge_densities)
        
        tampering_score = min(1.0, density_variance / (density_mean + 1e-6) / 2)
        
        threshold = density_mean + 2 * np.std(edge_densities)
        high_edge_regions = edge_densities > threshold
        
        return {
            'edge_density_mean': float(density_mean),
            'edge_density_variance': float(density_variance),
            'high_edge_region_percentage': float(np.sum(high_edge_regions) / high_edge_regions.size * 100),
            'tampering_score': tampering_score
        }


class SplicingDetector:
    """Combined splicing detector using multiple forensic techniques."""
    
    def __init__(self) -> None:
        self.ghost_detector = JPEGGhostDetector()
        self.dct_analyzer = DCTAnalyzer()
        self.edge_detector = EdgeIrregularityDetector()
    
    def analyze(self, image_path: str) -> dict[str, Any]:
        """Run all splicing detection methods and return combined results."""
        print(f"  Analyzing: {Path(image_path).name}")
        
        ghost_result = self.ghost_detector.analyze(image_path)
        dct_result = self.dct_analyzer.analyze(image_path)
        edge_result = self.edge_detector.analyze(image_path)
        
        weights = {'ghost': 0.45, 'dct': 0.35, 'edge': 0.20}
        
        ensemble_score = (
            weights['ghost'] * ghost_result['tampering_score'] +
            weights['dct'] * dct_result['tampering_score'] +
            weights['edge'] * edge_result['tampering_score']
        )
        
        if ensemble_score > 0.50:
            verdict = "TAMPERED"
            confidence = "HIGH" if ensemble_score > 0.65 else "MEDIUM"
        elif ensemble_score > 0.35:
            verdict = "POSSIBLY TAMPERED"
            confidence = "LOW"
        else:
            verdict = "AUTHENTIC"
            confidence = "HIGH" if ensemble_score < 0.20 else "MEDIUM"
        
        return {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'ghost_score': ghost_result['tampering_score'],
            'ghost_variance': ghost_result['quality_variance'],
            'ghost_suspicious_pct': ghost_result['suspicious_percentage'],
            'dct_score': dct_result['tampering_score'],
            'dct_inconsistency': dct_result['inconsistency_score'],
            'edge_score': edge_result['tampering_score'],
            'edge_density_var': edge_result['edge_density_variance'],
            'ensemble_score': ensemble_score,
            'verdict': verdict,
            'confidence': confidence
        }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python splicing_detector.py <image_path>")
        sys.exit(1)
    
    detector = SplicingDetector()
    result = detector.analyze(sys.argv[1])
    
    print(f"\nSplicing Analysis Results:")
    print(f"  Image: {result['image_name']}")
    print(f"  Ghost Score: {result['ghost_score']:.3f}")
    print(f"  DCT Score: {result['dct_score']:.3f}")
    print(f"  Edge Score: {result['edge_score']:.3f}")
    print(f"  Ensemble: {result['ensemble_score']:.3f}")
    print(f"  Verdict: {result['verdict']} ({result['confidence']} confidence)")

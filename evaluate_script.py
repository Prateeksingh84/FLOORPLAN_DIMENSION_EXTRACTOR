"""
Evaluation Script for Floorplan Dimension Extractor
This script computes Precision, Recall, and F1-score by comparing
extracted results against ground truth.
"""

import json
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ResultEvaluator:
    """Evaluate extraction results against ground truth"""
    
    def __init__(self, extracted_path: str, ground_truth_path: str):
        self.extracted_path = extracted_path
        self.ground_truth_path = ground_truth_path
        self.extracted_data = None
        self.ground_truth_data = None
    
    def load_data(self):
        """Load extracted results and ground truth"""
        try:
            with open(self.extracted_path, 'r') as f:
                self.extracted_data = json.load(f)
            logger.info(f"Loaded extracted results from {self.extracted_path}")
        except FileNotFoundError:
            logger.error(f"Extracted results file not found: {self.extracted_path}")
            return False
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in extracted results: {self.extracted_path}")
            return False
        
        try:
            with open(self.ground_truth_path, 'r') as f:
                self.ground_truth_data = json.load(f)
            logger.info(f"Loaded ground truth from {self.ground_truth_path}")
        except FileNotFoundError:
            logger.error(f"Ground truth file not found: {self.ground_truth_path}")
            return False
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in ground truth: {self.ground_truth_path}")
            return False
        
        return True
    
    def normalize_dimension(self, dim: Dict) -> Tuple:
        """Create normalized tuple for comparison"""
        # Use inches value rounded to 1 decimal for comparison
        # This allows for small floating point differences
        return (round(dim['inches'], 1),)
    
    def normalize_code(self, code: str) -> str:
        """Normalize cabinet code for comparison"""
        return code.upper().strip()
    
    def evaluate_dimensions(self) -> Dict:
        """Calculate metrics for dimension extraction"""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for page_idx, page in enumerate(self.ground_truth_data):
            if page_idx >= len(self.extracted_data):
                # Page exists in ground truth but not in extracted
                false_negatives += len(page['dimensions'])
                continue
            
            # Create sets of normalized dimensions
            gt_dims = set(self.normalize_dimension(d) for d in page['dimensions'])
            ext_dims = set(self.normalize_dimension(d) for d in self.extracted_data[page_idx]['dimensions'])
            
            # Calculate metrics
            tp = len(gt_dims & ext_dims)  # Intersection
            fp = len(ext_dims - gt_dims)   # Extracted but not in ground truth
            fn = len(gt_dims - ext_dims)   # In ground truth but not extracted
            
            true_positives += tp
            false_positives += fp
            false_negatives += fn
        
        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4)
        }
    
    def evaluate_codes(self) -> Dict:
        """Calculate metrics for cabinet code extraction"""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for page_idx, page in enumerate(self.ground_truth_data):
            if page_idx >= len(self.extracted_data):
                # Page exists in ground truth but not in extracted
                false_negatives += len(page['codes'])
                continue
            
            # Create sets of normalized codes
            gt_codes = set(self.normalize_code(c) for c in page['codes'])
            ext_codes = set(self.normalize_code(c) for c in self.extracted_data[page_idx]['codes'])
            
            # Calculate metrics
            tp = len(gt_codes & ext_codes)
            fp = len(ext_codes - gt_codes)
            fn = len(gt_codes - ext_codes)
            
            true_positives += tp
            false_positives += fp
            false_negatives += fn
        
        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4)
        }
    
    def print_report(self, dim_metrics: Dict, code_metrics: Dict):
        """Print detailed evaluation report"""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print("\n📏 DIMENSION EXTRACTION METRICS")
        print("-" * 60)
        print(f"True Positives:     {dim_metrics['true_positives']}")
        print(f"False Positives:    {dim_metrics['false_positives']}")
        print(f"False Negatives:    {dim_metrics['false_negatives']}")
        print(f"\nPrecision:          {dim_metrics['precision']:.2%}")
        print(f"Recall:             {dim_metrics['recall']:.2%}")
        print(f"F1-Score:           {dim_metrics['f1_score']:.2%}")
        
        print("\n🏷️  CABINET CODE EXTRACTION METRICS")
        print("-" * 60)
        print(f"True Positives:     {code_metrics['true_positives']}")
        print(f"False Positives:    {code_metrics['false_positives']}")
        print(f"False Negatives:    {code_metrics['false_negatives']}")
        print(f"\nPrecision:          {code_metrics['precision']:.2%}")
        print(f"Recall:             {code_metrics['recall']:.2%}")
        print(f"F1-Score:           {code_metrics['f1_score']:.2%}")
        
        print("\n" + "="*60)
        
        # Overall assessment
        avg_f1 = (dim_metrics['f1_score'] + code_metrics['f1_score']) / 2
        print(f"\n📊 OVERALL F1-SCORE: {avg_f1:.2%}")
        
        if avg_f1 >= 0.95:
            print("✅ Excellent performance!")
        elif avg_f1 >= 0.85:
            print("✅ Good performance!")
        elif avg_f1 >= 0.70:
            print("⚠️  Acceptable performance - room for improvement")
        else:
            print("❌ Poor performance - needs significant improvement")
        
        print("="*60 + "\n")
    
    def save_report(self, dim_metrics: Dict, code_metrics: Dict, output_path: str = "evaluation_report.json"):
        """Save evaluation report to JSON"""
        report = {
            "dimensions": dim_metrics,
            "codes": code_metrics,
            "overall_f1_score": round((dim_metrics['f1_score'] + code_metrics['f1_score']) / 2, 4)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def run_evaluation(self):
        """Run complete evaluation"""
        if not self.load_data():
            return
        
        dim_metrics = self.evaluate_dimensions()
        code_metrics = self.evaluate_codes()
        
        self.print_report(dim_metrics, code_metrics)
        self.save_report(dim_metrics, code_metrics)


def create_sample_ground_truth():
    """Create a sample ground truth file for testing"""
    sample_ground_truth = [
        {
            "page": 1,
            "dimensions": [
                {"raw": "34 1/2\"", "inches": 34.5, "bbox": [100, 200, 150, 220]},
                {"raw": "25\"", "inches": 25.0, "bbox": [200, 300, 230, 320]},
                {"raw": "2' 6\"", "inches": 30.0, "bbox": [300, 400, 350, 420]}
            ],
            "codes": ["DB24", "SB42FH", "W3030"]
        }
    ]
    
    with open("ground_truth_sample.json", 'w') as f:
        json.dump(sample_ground_truth, f, indent=2)
    
    logger.info("Sample ground truth created: ground_truth_sample.json")
    logger.info("Please update this file with your actual ground truth data")


def main():
    """Main execution"""
    import sys
    
    print("\n" + "="*60)
    print("FLOORPLAN EXTRACTOR - EVALUATION TOOL")
    print("="*60 + "\n")
    
    # Check if ground truth exists
    import os
    if not os.path.exists("ground_truth.json"):
        logger.warning("ground_truth.json not found")
        logger.info("Creating sample ground truth file...")
        create_sample_ground_truth()
        logger.info("\nPlease:")
        logger.info("1. Edit 'ground_truth_sample.json' with correct values")
        logger.info("2. Rename it to 'ground_truth.json'")
        logger.info("3. Run this script again")
        return
    
    if not os.path.exists("extracted_results.json"):
        logger.error("extracted_results.json not found")
        logger.info("Please run floorplan_extractor.py first to generate results")
        return
    
    # Run evaluation
    evaluator = ResultEvaluator("extracted_results.json", "ground_truth.json")
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
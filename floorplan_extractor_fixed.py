"""
Floorplan Dimension Extractor
McKH Technologies Assignment

This script extracts dimensions and cabinet codes from floorplan PDFs.
"""

import fitz  # PyMuPDF
import re
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np


class FloorplanExtractor:
    def __init__(self, pdf_path: str):
        """Initialize the extractor with a PDF file path."""
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

        # Regex patterns for dimension detection
        self.dimension_patterns = [
            # Matches: 34 1/2", 2 3/4", etc.
            r'\b(\d+)\s+(\d+)/(\d+)\s*["\']',
            # Matches: 34.5", 2.75", etc.
            r'\b(\d+\.?\d*)\s*["\']',
            # Matches: 2' 6", 3'4", etc.
            r'\b(\d+)\s*\'\s*(\d+\.?\d*)\s*["\']?',
            # Matches: 2'6", 3'4 1/2", etc.
            r'\b(\d+)\s*\'\s*(\d+)\s+(\d+)/(\d+)\s*["\']?',
        ]

        # Regex pattern for cabinet/appliance codes
        # Matches codes like: DB24, SB42FH, W3630, etc.
        self.code_pattern = r'\b([A-Z]{1,4}\d{2,4}[A-Z]{0,4})\b'

    def parse_dimension(self, text: str) -> Optional[float]:
        """
        Parse dimension string and convert to inches.
        """
        text = text.strip()

        # Pattern 1: X Y/Z" (e.g., 34 1/2")
        match = re.match(r'(\d+)\s+(\d+)/(\d+)\s*["\']', text)
        if match:
            whole = int(match.group(1))
            numerator = int(match.group(2))
            denominator = int(match.group(3))
            return whole + (numerator / denominator)

        # Pattern 2: X' Y" (e.g., 2' 6")
        match = re.match(r'(\d+)\s*\'\s*(\d+\.?\d*)\s*["\']?', text)
        if match:
            feet = int(match.group(1))
            inches = float(match.group(2)) if match.group(2) else 0
            return feet * 12 + inches

        # Pattern 3: X' Y Z/W" (e.g., 2' 6 1/2")
        match = re.match(r'(\d+)\s*\'\s*(\d+)\s+(\d+)/(\d+)\s*["\']?', text)
        if match:
            feet = int(match.group(1))
            inches_whole = int(match.group(2))
            numerator = int(match.group(3))
            denominator = int(match.group(4))
            return feet * 12 + inches_whole + (numerator / denominator)

        # Pattern 4: X" or X.Y" (e.g., 25", 34.5")
        match = re.match(r'(\d+\.?\d*)\s*["\']', text)
        if match:
            return float(match.group(1))

        return None

    def extract_dimensions_from_text(self, text: str, bbox: Tuple[float, float, float, float]) -> List[Dict]:
        """Extract all dimensions from a text block."""
        dimensions = []

        for pattern in self.dimension_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                raw_dim = match.group(0)
                inches = self.parse_dimension(raw_dim)
                if inches is not None:
                    dimensions.append({
                        "raw": raw_dim,
                        "inches": round(inches, 2),
                        "bbox": list(bbox)
                    })

        return dimensions

    def extract_codes_from_text(self, text: str) -> List[str]:
        """Extract cabinet/appliance codes from text."""
        matches = re.findall(self.code_pattern, text)
        return list(set(matches))  # Remove duplicates

    def extract_from_page(self, page_num: int) -> Dict:
        """Extract dimensions and codes from a single page."""
        page = self.doc[page_num]
        text_blocks = page.get_text("dict")["blocks"]

        dimensions = []
        codes = []

        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        bbox = span["bbox"]

                        dims = self.extract_dimensions_from_text(text, bbox)
                        dimensions.extend(dims)

                        found_codes = self.extract_codes_from_text(text)
                        codes.extend(found_codes)

        codes = list(set(codes))
        return {
            "page": page_num + 1,
            "dimensions": dimensions,
            "codes": sorted(codes)
        }

    def extract_all(self) -> List[Dict]:
        """Extract dimensions and codes from all pages."""
        results = []
        for page_num in range(len(self.doc)):
            page_result = self.extract_from_page(page_num)
            results.append(page_result)
        return results

    def visualize_page(self, page_num: int, output_path: str):
        """Visualize extracted dimensions on the PDF page."""
        page = self.doc[page_num]
        page_data = self.extract_from_page(page_num)

        # Convert PDF page to image
        mat = fitz.Matrix(2, 2)  # 2x zoom
        pix = page.get_pixmap(matrix=mat)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # 🔧 Fix: make array writable
        img = img.copy()

        # Convert RGBA/Gray to BGR
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Draw bounding boxes and labels
        for dim in page_data["dimensions"]:
            bbox = dim["bbox"]
            x0, y0, x1, y1 = [int(coord * 2) for coord in bbox]

            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

            label = f"{dim['raw']} ({dim['inches']}\")"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(
                img,
                (x0, y0 - text_height - 10),
                (x0 + text_width, y0),
                (0, 255, 0),
                -1
            )

            cv2.putText(img, label, (x0, y0 - 5), font, font_scale, (0, 0, 0), thickness)

        cv2.imwrite(output_path, img)
        print(f"Visualization saved to: {output_path}")

    def close(self):
        """Close the PDF document."""
        self.doc.close()


def calculate_metrics(predicted: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Calculate precision, recall, and F1-score."""
    pred_dims = set(dim['inches'] for dim in predicted)
    true_dims = set(dim['inches'] for dim in ground_truth)

    true_positives = len(pred_dims.intersection(true_dims))
    false_positives = len(pred_dims - true_dims)
    false_negatives = len(true_dims - pred_dims)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def main():
    PDF_PATH = "sample_floorplan.pdf"
    OUTPUT_JSON = "extracted_results.json"
    OUTPUT_DIR = "visualizations"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        print("Please run 'python generate_sample_pdf.py' first to create the sample PDF.")
        return

    print(f"Processing: {PDF_PATH}")
    print("=" * 60)

    extractor = FloorplanExtractor(PDF_PATH)

    print("\nExtracting dimensions and codes...")
    results = extractor.extract_all()

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_JSON}")

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)

    for page_result in results:
        page_num = page_result['page']
        num_dims = len(page_result['dimensions'])
        num_codes = len(page_result['codes'])

        print(f"\nPage {page_num}:")
        print(f"  Dimensions found: {num_dims}")
        print(f"  Codes found: {num_codes}")

        if page_result['dimensions']:
            print("  Sample dimensions:")
            for dim in page_result['dimensions'][:5]:
                print(f"    {dim['raw']} = {dim['inches']}\"")

        if page_result['codes']:
            print(f"  Codes: {', '.join(page_result['codes'][:10])}")

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    for page_num in range(len(extractor.doc)):
        output_img = os.path.join(OUTPUT_DIR, f"page_{page_num + 1}_annotated.png")
        print(f"\nProcessing page {page_num + 1}...")
        extractor.visualize_page(page_num, output_img)

    extractor.close()

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - JSON results: {OUTPUT_JSON}")
    print(f"  - Visualizations: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

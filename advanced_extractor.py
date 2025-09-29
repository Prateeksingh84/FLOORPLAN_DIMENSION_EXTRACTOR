"""
Advanced Floorplan Dimension Extractor with OCR Support
Standalone version - no external imports needed
"""

import fitz  # PyMuPDF
import re
import json
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Try to import pytesseract (optional)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AdvancedDimensionExtractor:
    """Advanced extractor with OCR support and standard text extraction"""
    
    def __init__(self, pdf_path: str, use_ocr: bool = False):
        self.pdf_path = pdf_path
        self.use_ocr = use_ocr and TESSERACT_AVAILABLE
        self.doc = fitz.open(pdf_path)
        self.results = []
        
        # Enhanced dimension patterns
        self.dimension_patterns = [
            r"(\d+)'\s*-?\s*(\d+)(?:\s+(\d+)/(\d+))?\"?",  # Feet-inches with dash
            r"(\d+)'\s*(\d+)(?:\s+(\d+)/(\d+))?\"?",       # Standard feet-inches
            r"(\d+)\s+(\d+)/(\d+)\"",                       # Fraction inches
            r"(\d+\.?\d*)\"",                               # Decimal inches
            r"(\d+)\s*mm",                                  # Millimeters
            r"(\d+)\s*cm",                                  # Centimeters
        ]
        
        # Enhanced code pattern
        self.code_pattern = r"\b([A-Z]{1,4}\d{2,4}[A-Z]{0,4})\b"
        
        if self.use_ocr and not TESSERACT_AVAILABLE:
            logger.warning("OCR requested but pytesseract not installed. Using text extraction only.")
            self.use_ocr = False
        
        if self.use_ocr:
            logger.info("OCR mode enabled")
    
    def convert_to_inches(self, match_text: str) -> Optional[float]:
        """Convert dimension string to float inches"""
        try:
            # Handle millimeters
            mm_match = re.match(r"(\d+)\s*mm", match_text, re.IGNORECASE)
            if mm_match:
                return round(float(mm_match.group(1)) / 25.4, 2)
            
            # Handle centimeters
            cm_match = re.match(r"(\d+)\s*cm", match_text, re.IGNORECASE)
            if cm_match:
                return round(float(cm_match.group(1)) / 2.54, 2)
            
            # Feet and inches with optional dash
            feet_inches = re.match(r"(\d+)'\s*-?\s*(\d+)(?:\s+(\d+)/(\d+))?\"?", match_text)
            if feet_inches:
                feet = int(feet_inches.group(1))
                inches = int(feet_inches.group(2))
                total = feet * 12 + inches
                if feet_inches.group(3) and feet_inches.group(4):
                    numerator = int(feet_inches.group(3))
                    denominator = int(feet_inches.group(4))
                    total += numerator / denominator
                return round(total, 2)
            
            # Inches with fraction
            inch_fraction = re.match(r"(\d+)\s+(\d+)/(\d+)\"", match_text)
            if inch_fraction:
                whole = int(inch_fraction.group(1))
                numerator = int(inch_fraction.group(2))
                denominator = int(inch_fraction.group(3))
                return round(whole + numerator / denominator, 2)
            
            # Plain or decimal inches
            plain_inches = re.match(r"(\d+\.?\d*)\"", match_text)
            if plain_inches:
                return round(float(plain_inches.group(1)), 2)
            
            return None
        except Exception as e:
            logger.debug(f"Error converting '{match_text}': {e}")
            return None
    
    def extract_text_from_page(self, page_num: int) -> Dict:
        """Extract dimensions using standard text extraction"""
        page = self.doc[page_num]
        text_blocks = page.get_text("dict")["blocks"]
        
        dimensions = []
        codes = set()
        
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        bbox = span["bbox"]
                        
                        # Extract dimensions
                        for pattern in self.dimension_patterns:
                            matches = re.finditer(pattern, text)
                            for match in matches:
                                inches = self.convert_to_inches(match.group())
                                if inches:
                                    dimensions.append({
                                        "raw": match.group(),
                                        "inches": inches,
                                        "bbox": [round(x, 2) for x in bbox]
                                    })
                        
                        # Extract codes
                        code_matches = re.finditer(self.code_pattern, text)
                        for code_match in code_matches:
                            codes.add(code_match.group(1))
        
        return {
            "page": page_num + 1,
            "dimensions": dimensions,
            "codes": sorted(list(codes))
        }
    
    def extract_with_ocr(self, page_num: int) -> Dict:
        """Extract text using OCR from PDF page"""
        if not self.use_ocr:
            return {"page": page_num + 1, "dimensions": [], "codes": []}
        
        try:
            page = self.doc[page_num]
            
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # High resolution
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(
                thresh, output_type=pytesseract.Output.DICT
            )
            
            dimensions = []
            codes = set()
            
            # Process OCR results
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                if not text:
                    continue
                
                conf = int(ocr_data['conf'][i])
                if conf < 30:  # Skip low confidence
                    continue
                
                x, y, w, h = (
                    ocr_data['left'][i], ocr_data['top'][i],
                    ocr_data['width'][i], ocr_data['height'][i]
                )
                bbox = [x / 3, y / 3, (x + w) / 3, (y + h) / 3]  # Scale back
                
                # Check for dimensions
                for pattern in self.dimension_patterns:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        inches = self.convert_to_inches(match.group())
                        if inches:
                            dimensions.append({
                                "raw": match.group(),
                                "inches": inches,
                                "bbox": [round(x, 2) for x in bbox],
                                "confidence": conf
                            })
                
                # Check for codes
                code_matches = re.finditer(self.code_pattern, text)
                for code_match in code_matches:
                    codes.add(code_match.group(1))
            
            return {
                "page": page_num + 1,
                "dimensions": dimensions,
                "codes": sorted(list(codes))
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed for page {page_num + 1}: {e}")
            return {"page": page_num + 1, "dimensions": [], "codes": []}
    
    def extract_hybrid(self) -> List[Dict]:
        """Use both text extraction and OCR, then merge"""
        logger.info("Starting hybrid extraction (Text + OCR)...")
        
        merged = []
        
        for page_num in range(len(self.doc)):
            # Get text extraction results
            text_result = self.extract_text_from_page(page_num)
            
            # Get OCR results if enabled
            ocr_result = self.extract_with_ocr(page_num) if self.use_ocr else None
            
            # Merge results
            page_data = {"page": page_num + 1, "dimensions": [], "codes": []}
            
            # Merge dimensions
            dim_set = set()
            
            for dim in text_result["dimensions"]:
                key = (dim["raw"], dim["inches"])
                if key not in dim_set:
                    page_data["dimensions"].append(dim)
                    dim_set.add(key)
            
            if ocr_result:
                for dim in ocr_result["dimensions"]:
                    key = (dim["raw"], dim["inches"])
                    if key not in dim_set:
                        page_data["dimensions"].append(dim)
                        dim_set.add(key)
            
            # Merge codes
            codes_set = set(text_result["codes"])
            if ocr_result:
                codes_set.update(ocr_result["codes"])
            
            page_data["codes"] = sorted(list(codes_set))
            merged.append(page_data)
        
        logger.info(f"Extraction complete: {len(merged)} pages")
        return merged
    
    def extract(self) -> List[Dict]:
        """Main extraction method"""
        if self.use_ocr:
            self.results = self.extract_hybrid()
        else:
            logger.info("Starting text extraction...")
            self.results = []
            for page_num in range(len(self.doc)):
                result = self.extract_text_from_page(page_num)
                self.results.append(result)
        
        return self.results
    
    def visualize_page(self, page_num: int, output_path: str):
        """Visualize extracted dimensions on the PDF page"""
        page = self.doc[page_num]
        
        # Get page results
        if page_num < len(self.results):
            page_data = self.results[page_num]
        else:
            page_data = self.extract_text_from_page(page_num)
        
        # Convert PDF page to image
        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # Convert RGBA to BGR for OpenCV
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding boxes and labels
        for dim in page_data["dimensions"]:
            bbox = dim["bbox"]
            # Scale bbox coordinates to match the zoomed image
            x0, y0, x1, y1 = [int(coord * 2) for coord in bbox]
            
            # Draw rectangle
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            
            # Draw label with dimension
            label = f"{dim['raw']} ({dim['inches']}\")"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                img,
                (x0, y0 - text_height - 10),
                (x0 + text_width, y0),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                img,
                label,
                (x0, y0 - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )
        
        # Save image
        cv2.imwrite(output_path, img)
        logger.info(f"Visualization saved to: {output_path}")
    
    def save_json(self, output_path: str):
        """Save results to JSON"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
    
    def close(self):
        """Close the PDF document"""
        self.doc.close()


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Floorplan Dimension Extractor')
    parser.add_argument('--pdf', default='sample_floorplan.pdf', help='Path to PDF file')
    parser.add_argument('--ocr', action='store_true', help='Enable OCR mode')
    parser.add_argument('--output', default='extracted_results.json', help='Output JSON file')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization images')
    parser.add_argument('--viz-dir', default='visualizations', help='Visualization output directory')
    
    args = parser.parse_args()
    
    if not Path(args.pdf).exists():
        logger.error(f"PDF file not found: {args.pdf}")
        logger.info("Run 'python generate_sample_pdf.py' to create a sample PDF first.")
        return 1
    
    if args.ocr and not TESSERACT_AVAILABLE:
        logger.error("OCR requested but pytesseract not installed")
        logger.info("Install with: pip install pytesseract")
        logger.info("Also install Tesseract: https://github.com/tesseract-ocr/tesseract")
        return 1
    
    # Extract
    extractor = AdvancedDimensionExtractor(args.pdf, use_ocr=args.ocr)
    results = extractor.extract()
    extractor.save_json(args.output)
    
    # Generate visualizations if requested
    if args.visualize:
        os.makedirs(args.viz_dir, exist_ok=True)
        logger.info(f"Generating visualizations in {args.viz_dir}/")
        
        for page_num in range(len(extractor.doc)):
            output_img = os.path.join(args.viz_dir, f"page_{page_num + 1}_annotated.png")
            extractor.visualize_page(page_num, output_img)
    
    extractor.close()
    
    # Summary
    total_dims = sum(len(page["dimensions"]) for page in results)
    total_codes = sum(len(page["codes"]) for page in results)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"EXTRACTION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Pages processed: {len(results)}")
    logger.info(f"Total dimensions found: {total_dims}")
    logger.info(f"Total codes found: {total_codes}")
    
    for page in results:
        if page["dimensions"] or page["codes"]:
            logger.info(f"\nPage {page['page']}:")
            logger.info(f"  Dimensions: {len(page['dimensions'])}")
            logger.info(f"  Codes: {len(page['codes'])}")
            if page["dimensions"][:3]:
                logger.info(f"  Sample: {[d['raw'] for d in page['dimensions'][:3]]}")
    
    logger.info(f"{'='*50}\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
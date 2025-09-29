# Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Install Dependencies (1 minute)
```bash
python setup.py
```

This will automatically:
- Check Python version
- Install all required packages
- Create necessary directories
- Verify installation

### Step 2: Add Your PDF (30 seconds)
```bash
# Copy your floorplan PDF to the project directory
cp /path/to/your/floorplan.pdf floorplan.pdf
```

### Step 3: Run the Extractor (1 minute)
```bash
python floorplan_extractor.py
```

**That's it!** Your results are ready:
- `extracted_results.json` - All extracted data
- `annotated_images/` - Visualized PDFs with bounding boxes

---

## 📁 File Structure

```
floorplan-extractor/
├── floorplan_extractor.py      # Main extraction script
├── advanced_extractor.py        # Version with OCR support
├── evaluate_results.py          # Evaluation metrics tool
├── setup.py                     # One-click setup
├── requirements.txt             # Python dependencies
├── README.md                    # Full documentation
├── QUICK_START.md              # This file
├── config.py                    # Configuration (auto-generated)
├── floorplan_extractor.code-workspace  # VS Code workspace
│
├── floorplan.pdf               # Your input PDF (add this)
├── extracted_results.json      # Output data (generated)
├── ground_truth.json           # For evaluation (optional)
│
├── annotated_images/           # Visualizations (generated)
│   ├── page_1_annotated.png
│   ├── page_2_annotated.png
│   └── ...
│
└── sample_pdfs/                # Place test PDFs here
```

---

## 💡 Common Use Cases

### Basic Extraction
```bash
python floorplan_extractor.py
```

### With OCR (for scanned PDFs)
```bash
# Install OCR support first
pip install pytesseract

# Run with OCR
python advanced_extractor.py floorplan.pdf --ocr
```

### Custom Output Location
Edit line 232 in `floorplan_extractor.py`:
```python
PDF_PATH = "my_custom_floorplan.pdf"
JSON_OUTPUT = "my_results.json"
IMAGE_OUTPUT_DIR = "my_images"
```

### Batch Process Multiple PDFs
```bash
python batch_process.py sample_pdfs/
```

---

## 🔧 Troubleshooting

### Problem: No dimensions detected
**Solutions:**
1. Check if PDF has selectable text (try copying text from PDF)
2. If scanned PDF, use OCR mode: `python advanced_extractor.py --ocr floorplan.pdf`
3. Verify dimensions include quote marks (e.g., `25"` not just `25`)

### Problem: ModuleNotFoundError
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### Problem: Low accuracy
**Try:**
1. Use hybrid extraction (enabled by default)
2. Enable OCR mode for scanned PDFs
3. Check ground truth format in evaluation

### Problem: Images not generated
```bash
# Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# Reinstall if needed
pip install opencv-python --upgrade
```

---

## 📊 Example Output

### JSON Format:
```json
{
  "page": 1,
  "dimensions": [
    {
      "raw": "34 1/2\"",
      "inches": 34.5,
      "bbox": [150.5, 200.3, 180.7, 215.9]
    },
    {
      "raw": "2' 6\"",
      "inches": 30.0,
      "bbox": [250.1, 300.5, 290.3, 320.8]
    }
  ],
  "codes": ["DB24", "SB42FH", "W3030"]
}
```

### Annotated Images:
- Green boxes around detected dimensions
- Red labels showing: `34 1/2" (34.5")`
- High resolution (2x zoom)

---

## 🎯 Evaluation (Optional)

### Step 1: Create Ground Truth
```bash
python evaluate_results.py
# This creates ground_truth_sample.
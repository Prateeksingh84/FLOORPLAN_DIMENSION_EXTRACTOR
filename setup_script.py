"""
Setup script for Floorplan Dimension Extractor
Installs all dependencies and sets up the project
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    dirs = ["annotated_images", "output_images", "sample_pdfs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created: {dir_name}/")
    return True


def verify_installation():
    """Verify all packages are installed correctly"""
    print("\n🔍 Verifying installation...")
    required_packages = {
        'fitz': 'PyMuPDF',
        'pdfplumber': 'pdfplumber',
        'cv2': 'opencv-python',
        'numpy': 'numpy'
    }
    
    all_installed = True
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package} imported successfully")
        except ImportError:
            print(f"❌ {package} import failed")
            all_installed = False
    
    return all_installed


def create_sample_config():
    """Create a sample configuration file"""
    print("\n⚙️  Creating sample configuration...")
    config = """# Floorplan Extractor Configuration
# Edit these paths as needed

PDF_PATH = "floorplan.pdf"
JSON_OUTPUT = "extracted_results.json"
IMAGE_OUTPUT_DIR = "annotated_images"

# Visualization settings
BBOX_COLOR = (0, 255, 0)  # Green
LABEL_COLOR = (0, 0, 255)  # Red
ZOOM_FACTOR = 2  # Image resolution multiplier

# Extraction settings
MERGE_RESULTS = True  # Merge PyMuPDF and pdfplumber results
MIN_CONFIDENCE = 0.8  # Minimum confidence for detection (not used yet)
"""
    
    with open("config.py", "w") as f:
        f.write(config)
    print("✅ Configuration file created: config.py")
    return True


def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📝 NEXT STEPS:")
    print("\n1. Place your floorplan PDF in this directory:")
    print("   - Name it 'floorplan.pdf'")
    print("   - Or update PDF_PATH in the script")
    
    print("\n2. Run the extractor:")
    print("   python floorplan_extractor.py")
    
    print("\n3. Check the outputs:")
    print("   - extracted_results.json (dimension data)")
    print("   - annotated_images/ (visualizations)")
    
    print("\n4. (Optional) Run evaluation:")
    print("   - Create ground_truth.json")
    print("   - Run: python evaluate_results.py")
    
    print("\n5. Open in VS Code:")
    print("   code floorplan_extractor.code-workspace")
    
    print("\n📚 Documentation: See README.md for detailed instructions")
    print("="*60 + "\n")


def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("FLOORPLAN DIMENSION EXTRACTOR - SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Create directories
    if not create_directories():
        return 1
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages failed to import. Try:")
        print("   pip install --force-reinstall -r requirements.txt")
        return 1
    
    # Create sample config
    create_sample_config()
    
    # Print next steps
    print_next_steps()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

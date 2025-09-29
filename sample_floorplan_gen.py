"""
Generate a sample floorplan PDF with dimensions and cabinet codes for testing.
This creates a realistic floorplan that the extractor can process.
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import os

def create_sample_floorplan(output_path="sample_floorplan.pdf"):
    """Create a sample floorplan PDF with dimensions and cabinet codes."""
    
    # Create PDF
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "KITCHEN FLOORPLAN")
    
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, "Sample Floor Plan - Test Document")
    
    # Draw kitchen outline
    kitchen_x = 100
    kitchen_y = 200
    kitchen_width = 400
    kitchen_height = 300
    
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(2)
    c.rect(kitchen_x, kitchen_y, kitchen_width, kitchen_height)
    
    # Add overall dimensions
    c.setFont("Helvetica", 9)
    
    # Width dimension (top)
    c.drawString(kitchen_x + kitchen_width/2 - 20, kitchen_y + kitchen_height + 20, "16' 8\"")
    c.line(kitchen_x, kitchen_y + kitchen_height + 10, 
           kitchen_x + kitchen_width, kitchen_y + kitchen_height + 10)
    
    # Height dimension (right side)
    c.drawString(kitchen_x + kitchen_width + 20, kitchen_y + kitchen_height/2, "12' 6\"")
    c.line(kitchen_x + kitchen_width + 10, kitchen_y,
           kitchen_x + kitchen_width + 10, kitchen_y + kitchen_height)
    
    # Draw base cabinets along bottom wall
    c.setFont("Helvetica", 8)
    
    # Cabinet 1
    cab_y = kitchen_y + 10
    c.rect(kitchen_x + 20, cab_y, 60, 40)
    c.drawString(kitchen_x + 35, cab_y + 45, "24\"")
    c.drawString(kitchen_x + 30, cab_y + 15, "DB24")
    
    # Cabinet 2
    c.rect(kitchen_x + 90, cab_y, 80, 40)
    c.drawString(kitchen_x + 115, cab_y + 45, "36\"")
    c.drawString(kitchen_x + 110, cab_y + 15, "SB36")
    
    # Cabinet 3 - Sink base
    c.rect(kitchen_x + 180, cab_y, 90, 40)
    c.drawString(kitchen_x + 210, cab_y + 45, "42\"")
    c.drawString(kitchen_x + 200, cab_y + 15, "SB42FH")
    
    # Cabinet 4
    c.rect(kitchen_x + 280, cab_y, 55, 40)
    c.drawString(kitchen_x + 295, cab_y + 45, "21\"")
    c.drawString(kitchen_x + 290, cab_y + 15, "DB21")
    
    # Cabinet 5
    c.rect(kitchen_x + 345, cab_y, 40, 40)
    c.drawString(kitchen_x + 355, cab_y + 45, "15\"")
    c.drawString(kitchen_x + 350, cab_y + 15, "DB15")
    
    # Draw wall cabinets along top
    cab_top_y = kitchen_y + kitchen_height - 60
    
    # Wall cabinet 1
    c.rect(kitchen_x + 20, cab_top_y, 70, 50)
    c.drawString(kitchen_x + 40, cab_top_y - 10, "30\"")
    c.drawString(kitchen_x + 35, cab_top_y + 20, "W3030")
    
    # Wall cabinet 2
    c.rect(kitchen_x + 100, cab_top_y, 85, 50)
    c.drawString(kitchen_x + 125, cab_top_y - 10, "36\"")
    c.drawString(kitchen_x + 120, cab_top_y + 20, "W3636")
    
    # Wall cabinet 3
    c.rect(kitchen_x + 195, cab_top_y, 75, 50)
    c.drawString(kitchen_x + 215, cab_top_y - 10, "33\"")
    c.drawString(kitchen_x + 210, cab_top_y + 20, "W3336")
    
    # Wall cabinet 4
    c.rect(kitchen_x + 280, cab_top_y, 60, 50)
    c.drawString(kitchen_x + 295, cab_top_y - 10, "24\"")
    c.drawString(kitchen_x + 290, cab_top_y + 20, "W2430")
    
    # Draw appliances on right wall
    # Refrigerator
    c.setFillColorRGB(0.9, 0.9, 0.9)
    c.rect(kitchen_x + kitchen_width - 70, kitchen_y + 150, 60, 70, fill=1)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(kitchen_x + kitchen_width - 55, kitchen_y + 225, "36\"")
    c.drawString(kitchen_x + kitchen_width - 50, kitchen_y + 180, "REF36")
    
    # Dishwasher
    c.setFillColorRGB(0.9, 0.9, 0.9)
    c.rect(kitchen_x + 345, cab_y, 40, 40, fill=1)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(kitchen_x + 355, cab_y + 45, "24\"")
    c.drawString(kitchen_x + 350, cab_y + 15, "DW24")
    
    # Add some individual dimensions with fractions
    c.setFont("Helvetica", 7)
    c.drawString(kitchen_x + 50, kitchen_y - 15, "34 1/2\"")
    c.drawString(kitchen_x + 150, kitchen_y - 15, "2' 6\"")
    c.drawString(kitchen_x + 250, kitchen_y - 15, "18 3/4\"")
    
    # Add notes section
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, 150, "CABINET SCHEDULE:")
    
    c.setFont("Helvetica", 8)
    notes = [
        "DB24 - Drawer Base 24\"",
        "SB36 - Sink Base 36\"",
        "SB42FH - Sink Base 42\" with False Front",
        "W3030 - Wall Cabinet 30\" x 30\"",
        "W3636 - Wall Cabinet 36\" x 36\"",
        "DW24 - Dishwasher 24\"",
        "REF36 - Refrigerator Space 36\""
    ]
    
    y_pos = 130
    for note in notes:
        c.drawString(60, y_pos, note)
        y_pos -= 15
    
    # Add scale
    c.setFont("Helvetica", 8)
    c.drawString(50, 50, "SCALE: 1/4\" = 1'-0\"")
    c.drawString(50, 35, "DATE: 2024")
    
    # Save PDF
    c.save()
    print(f"Sample floorplan PDF created: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")


if __name__ == "__main__":
    # Generate the sample PDF
    output_file = "sample_floorplan.pdf"
    
    try:
        create_sample_floorplan(output_file)
        print("\nSuccess! You can now run the extractor on this file.")
        print(f"\nTo use it, make sure your extractor script has:")
        print(f'PDF_PATH = r"{os.path.abspath(output_file)}"')
    except Exception as e:
        print(f"Error creating PDF: {e}")
        print("\nMake sure you have reportlab installed:")
        print("pip install reportlab")

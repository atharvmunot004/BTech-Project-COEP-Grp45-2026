#!/usr/bin/env python3
"""
Convert ASCII art text file to PDF while preserving formatting.
Requires: pip install reportlab
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Preformatted, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttf import TTFont
import sys
import os

def convert_txt_to_pdf(input_file, output_file=None):
    """Convert text file to PDF with monospace font."""
    
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.pdf'
    
    # Read the text file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Container for the 'Flowable' objects
    story = []
    
    # Use monospace font - Courier is built-in
    # For better Unicode support, you might want to use a font like 'DejaVu Sans Mono'
    # but Courier works well for ASCII art
    style = getSampleStyleSheet()['Code']
    style.fontName = 'Courier'
    style.fontSize = 8  # Adjust size as needed
    style.leading = 9.6  # Line spacing
    
    # Add the preformatted text
    preformatted = Preformatted(
        content,
        style,
        maxLineLength=120,  # Adjust based on your content width
        splitLongWords=False,
        newLineChars='\n'
    )
    
    story.append(preformatted)
    
    # Build PDF
    doc.build(story)
    print(f"✓ PDF created successfully: {output_file}")

if __name__ == "__main__":
    input_file = "architecture.txt"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    convert_txt_to_pdf(input_file)


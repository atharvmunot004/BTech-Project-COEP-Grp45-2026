#!/usr/bin/env python3
"""
Convert .sty markdown file with LaTeX math to PDF.
This script converts markdown-style content with LaTeX equations to a proper LaTeX document and compiles it.
"""

import os
import sys
import re
import subprocess
from pathlib import Path

def markdown_to_latex(content):
    """Convert markdown-style content to LaTeX format."""
    
    # Start with LaTeX document structure
    latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{hyperref}
\geometry{margin=1in}

\title{Black-Litterman Model Explanation}
\author{}
\date{}

\begin{document}
\maketitle

"""
    
    lines = content.split('\n')
    i = 0
    
    def escape_latex(text):
        """Escape special LaTeX characters."""
        special_chars = ['&', '%', '$', '#', '^', '_', '{', '}', '~', '\\']
        for char in special_chars:
            if char == '\\':
                continue  # Don't escape backslashes that are part of LaTeX commands
            text = text.replace(char, '\\' + char)
        return text
    
    while i < len(lines):
        line = lines[i]
        
        # Handle multi-line LaTeX math blocks (lines starting with '[')
        if line.strip() == '[':
            math_content = []
            i += 1
            # Collect all lines until we find the closing ']'
            while i < len(lines) and lines[i].strip() != ']':
                math_content.append(lines[i])
                i += 1
            # Join the math content
            math_text = '\n'.join(math_content).strip()
            if math_text:
                latex += f"\\[\n{math_text}\n\\]\n\n"
            i += 1
            continue
        
        # Handle headers
        if line.startswith('# '):
            header_text = line[2:].strip()
            # Remove markdown bold and emojis for section title
            header_text = re.sub(r'\*\*(.*?)\*\*', r'\1', header_text)
            header_text = re.sub(r'[🟥🟩🟦🟧✔👉]', '', header_text).strip()
            latex += f"\\section{{{header_text}}}\n\n"
        elif line.startswith('## '):
            header_text = line[3:].strip()
            header_text = re.sub(r'\*\*(.*?)\*\*', r'\1', header_text)
            header_text = re.sub(r'[🟥🟩🟦🟧✔👉]', '', header_text).strip()
            latex += f"\\subsection{{{header_text}}}\n\n"
        elif line.startswith('### '):
            header_text = line[4:].strip()
            header_text = re.sub(r'\*\*(.*?)\*\*', r'\1', header_text)
            header_text = re.sub(r'[🟥🟩🟦🟧✔👉]', '', header_text).strip()
            latex += f"\\subsubsection{{{header_text}}}\n\n"
        
        # Handle horizontal rules
        elif line.strip() == '---':
            latex += "\\vspace{0.5cm}\n\\hrule\n\\vspace{0.5cm}\n\n"
        
        # Handle blockquotes
        elif line.startswith('> '):
            quote_text = line[2:].strip()
            # Remove markdown bold
            quote_text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', quote_text)
            latex += f"\\begin{{quote}}\n{quote_text}\n\\end{{quote}}\n\n"
        
        # Handle lists with inline math (e.g., "* ( \pi ) = ...")
        elif line.strip().startswith('* '):
            latex += "\\begin{itemize}\n"
            while i < len(lines) and lines[i].strip().startswith('* '):
                item_text = lines[i].strip()[2:]
                # Handle inline math in parentheses like "( \pi )" - remove spaces and backslashes are already there
                item_text = re.sub(r'\( (.*?) \)', r'$\1$', item_text)
                # Remove markdown bold
                item_text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', item_text)
                # Handle inline code
                item_text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', item_text)
                latex += f"\\item {item_text}\n"
                i += 1
            latex += "\\end{itemize}\n\n"
            continue
        
        # Handle numbered lists
        elif re.match(r'^\d+\.\s+', line.strip()):
            latex += "\\begin{enumerate}\n"
            while i < len(lines) and re.match(r'^\d+\.\s+', lines[i].strip()):
                item_text = re.sub(r'^\d+\.\s+', '', lines[i].strip())
                item_text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', item_text)
                latex += f"\\item {item_text}\n"
                i += 1
            latex += "\\end{enumerate}\n\n"
            continue
        
        # Handle tables (simple markdown tables)
        elif '|' in line and line.strip().startswith('|'):
            latex += "\\begin{table}[h]\n\\centering\n\\begin{tabular}{"
            # Count columns
            cols = [c.strip() for c in line.split('|') if c.strip()]
            num_cols = len(cols)
            latex += 'l' * num_cols + "}\n\\toprule\n"
            
            # Header row
            header = ' & '.join(cols)
            header = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', header)
            latex += f"{header} \\\\\n\\midrule\n"
            
            i += 1
            # Skip separator row (|---|---|)
            if i < len(lines) and '---' in lines[i]:
                i += 1
            
            # Data rows
            while i < len(lines) and '|' in lines[i] and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].split('|') if c.strip()]
                row_text = ' & '.join(row)
                row_text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', row_text)
                latex += f"{row_text} \\\\\n"
                i += 1
            
            latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
            continue
        
        # Handle regular text with markdown formatting
        elif line.strip():
            # Convert markdown bold to LaTeX
            text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', line)
            # Convert inline code
            text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)
            # Remove emojis (they won't render well in standard LaTeX)
            text = re.sub(r'[🟥🟩🟦🟧✔👉]', '', text)
            latex += f"{text}\n\n"
        
        # Empty lines
        else:
            latex += "\n"
        
        i += 1
    
    latex += "\\end{document}\n"
    return latex

def convert_sty_to_pdf(input_file, output_file=None, keep_tex=False):
    """Convert .sty file to PDF."""
    
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.pdf'
    
    # Read the input file
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert to LaTeX
    print("Converting to LaTeX...")
    latex_content = markdown_to_latex(content)
    
    # Write LaTeX file
    tex_file = os.path.splitext(input_file)[0] + '.tex'
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"LaTeX file created: {tex_file}")
    
    # Compile LaTeX to PDF
    print("Compiling LaTeX to PDF...")
    try:
        # Try pdflatex first
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_file],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            cwd=os.path.dirname(tex_file) or '.'
        )
        
        if result.returncode != 0:
            print("Warning: pdflatex had some issues. Trying xelatex for better Unicode support...")
            result = subprocess.run(
                ['xelatex', '-interaction=nonstopmode', tex_file],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                cwd=os.path.dirname(tex_file) or '.'
            )
        
        if result.returncode == 0:
            print(f"✓ PDF created successfully: {output_file}")
            if not keep_tex:
                # Clean up auxiliary files
                base_name = os.path.splitext(tex_file)[0]
                for ext in ['.aux', '.log', '.out']:
                    aux_file = base_name + ext
                    if os.path.exists(aux_file):
                        os.remove(aux_file)
                if not keep_tex:
                    if os.path.exists(tex_file):
                        os.remove(tex_file)
        else:
            print("Error compiling LaTeX:")
            print(result.stderr)
            print("\nLaTeX file saved for manual compilation:", tex_file)
            return False
            
    except FileNotFoundError:
        print("Error: LaTeX compiler (pdflatex or xelatex) not found.")
        print("Please install a LaTeX distribution (e.g., MiKTeX or TeX Live)")
        print(f"LaTeX file saved: {tex_file}")
        print("You can compile it manually with: pdflatex", tex_file)
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_sty_to_pdf.py <input.sty> [output.pdf]")
        print("Example: python convert_sty_to_pdf.py 012-Black-Litterman.sty")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    keep_tex = '--keep-tex' in sys.argv
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    convert_sty_to_pdf(input_file, output_file, keep_tex)


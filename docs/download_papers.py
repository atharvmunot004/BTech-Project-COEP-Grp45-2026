#!/usr/bin/env python3
"""
Script to download all papers from references_list_details.json
and save them to docs/reference-literature/
"""

import json
import os
import re
import time
import requests
from urllib.parse import urlparse
from pathlib import Path
import sys

# Configuration
JSON_FILE = "architecture-diagrams/references_list_details.json"
OUTPUT_DIR = "reference-literature"
DELAY_BETWEEN_REQUESTS = 2  # seconds
TIMEOUT = 30  # seconds

# User agent to avoid blocking
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def sanitize_filename(title, max_length=200):
    """Convert paper title to a safe filename"""
    # Remove special characters and replace with underscores
    filename = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace spaces and multiple underscores with single underscore
    filename = re.sub(r'[\s_]+', '_', filename)
    # Remove leading/trailing underscores and dots
    filename = filename.strip('._')
    # Truncate if too long
    if len(filename) > max_length:
        filename = filename[:max_length]
    return filename

def get_arxiv_pdf_url(url):
    """Convert arXiv abs URL to PDF URL"""
    if 'arxiv.org/abs/' in url:
        return url.replace('/abs/', '/pdf/') + '.pdf'
    elif 'arxiv.org/pdf/' in url:
        return url if url.endswith('.pdf') else url + '.pdf'
    return url

def get_filename_for_paper(paper, index):
    """Generate filename for a paper"""
    title = paper.get('title', f'Paper_{index}')
    year = paper.get('year', '')
    source = paper.get('source', '')
    
    # Create base filename
    filename = sanitize_filename(title)
    
    # Add year if available
    if year:
        filename = f"{year}_{filename}"
    
    # Try to determine extension from URL
    url = paper.get('url', '')
    if url:
        if 'arxiv.org' in url or url.endswith('.pdf'):
            extension = '.pdf'
        elif 'pdf' in url.lower():
            extension = '.pdf'
        else:
            extension = '.html'
    else:
        extension = '.txt'
    
    return filename + extension

def download_file(url, filepath):
    """Download a file from URL"""
    try:
        # Special handling for arXiv
        if 'arxiv.org' in url:
            pdf_url = get_arxiv_pdf_url(url)
            response = requests.get(pdf_url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(f"  Warning: Failed to download arXiv PDF, status {response.status_code}")
        
        # Special handling for ResearchGate - try to get PDF link
        if 'researchgate.net' in url:
            # ResearchGate usually requires login, save as HTML
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
            if response.status_code == 200:
                with open(filepath, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(response.text)
                return True
        
        # For other URLs, try direct download
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True, allow_redirects=True)
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '').lower()
            final_url = response.url.lower()
            
            # Check if it's a PDF
            is_pdf = ('pdf' in content_type or 
                     final_url.endswith('.pdf') or 
                     'application/pdf' in content_type)
            
            if is_pdf:
                # Save as PDF
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            else:
                # Save as HTML/text
                try:
                    response.encoding = response.apparent_encoding or 'utf-8'
                    with open(filepath, 'w', encoding='utf-8', errors='ignore') as f:
                        f.write(response.text)
                    return True
                except Exception as e:
                    # Fallback to binary if text fails
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    return True
        elif response.status_code == 403:
            print(f"  Warning: Access forbidden (403) - may require login/subscription")
            return False
        elif response.status_code == 404:
            print(f"  Warning: Page not found (404)")
            return False
        else:
            print(f"  Warning: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"  Error: Request timeout")
        return False
    except requests.exceptions.RequestException as e:
        print(f"  Error: {str(e)}")
        return False
    except Exception as e:
        print(f"  Unexpected error: {str(e)}")
        return False

def create_placeholder_file(filepath, paper):
    """Create a placeholder file when URL is missing"""
    content = f"""Paper Title: {paper.get('title', 'N/A')}
Authors: {paper.get('authors', 'N/A')}
Year: {paper.get('year', 'N/A')}
Source: {paper.get('source', 'N/A')}
URL: {paper.get('url', 'N/A')}

Note: This paper could not be downloaded automatically.
Please download it manually using the information above.
"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    # Get the script directory and set up paths
    script_dir = Path(__file__).parent
    json_path = script_dir / JSON_FILE
    output_dir = script_dir / OUTPUT_DIR
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JSON file
    print(f"Loading papers from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        sys.exit(1)
    
    papers = data.get('papers', [])
    total = len(papers)
    print(f"Found {total} papers to process.\n")
    
    # Track statistics
    downloaded = 0
    skipped = 0
    failed = 0
    
    # Process each paper
    for idx, paper in enumerate(papers, 1):
        title = paper.get('title', f'Paper_{idx}')
        url = paper.get('url')
        year = paper.get('year', '')
        
        print(f"[{idx}/{total}] {title[:60]}...")
        
        # Generate filename
        filename = get_filename_for_paper(paper, idx)
        filepath = output_dir / filename
        
        # Skip if file already exists
        if filepath.exists():
            print(f"  Already exists: {filename}")
            skipped += 1
            continue
        
        # Check if URL exists
        if not url:
            print(f"  No URL provided, creating placeholder file...")
            create_placeholder_file(filepath, paper)
            skipped += 1
            continue
        
        # Download the paper
        print(f"  Downloading from {url[:60]}...")
        success = download_file(url, filepath)
        
        if success:
            file_size = filepath.stat().st_size
            print(f"  [OK] Saved: {filename} ({file_size:,} bytes)")
            downloaded += 1
        else:
            print(f"  [FAILED] Could not download")
            failed += 1
            # Create placeholder file on failure
            create_placeholder_file(filepath, paper)
        
        # Be polite - wait between requests
        if idx < total:
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        print()
    
    # Print summary
    print("\n" + "="*60)
    print("Download Summary:")
    print(f"  Total papers: {total}")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Skipped (already exists/no URL): {skipped}")
    print(f"  Failed: {failed}")
    print(f"\nFiles saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()


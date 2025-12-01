#!/usr/bin/env python3
"""
Script to generate a report of downloaded vs failed papers
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Configuration
JSON_FILE = "architecture-diagrams/references_list_details.json"
OUTPUT_DIR = "reference-literature"
REPORT_FILE = "reference-literature/download_report.txt"

def sanitize_filename(title, max_length=200):
    """Convert paper title to a safe filename (same as download script)"""
    import re
    filename = re.sub(r'[<>:"/\\|?*]', '', title)
    filename = re.sub(r'[\s_]+', '_', filename)
    filename = filename.strip('._')
    if len(filename) > max_length:
        filename = filename[:max_length]
    return filename

def get_filename_for_paper(paper, index):
    """Generate filename for a paper (same as download script)"""
    title = paper.get('title', f'Paper_{index}')
    year = paper.get('year', '')
    
    filename = sanitize_filename(title)
    
    if year:
        filename = f"{year}_{filename}"
    
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

def check_file_type(filepath):
    """Check if file is a placeholder, error page, or actual download"""
    try:
        file_size = filepath.stat().st_size
        
        # Check for placeholder file
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(500)  # Read first 500 chars
            if 'This paper could not be downloaded automatically' in content:
                return ('placeholder', 'Placeholder file')
        
        # Check PDF files
        if filepath.suffix == '.pdf':
            with open(filepath, 'rb') as f:
                header = f.read(4)
                if header == b'%PDF':
                    return ('pdf', 'PDF document')
                else:
                    return ('invalid', 'Invalid PDF')
        
        # Check HTML files - small files likely error pages
        elif filepath.suffix in ['.html', '.txt']:
            # Check for common error indicators
            error_indicators = [
                '403', '404', 'Forbidden', 'Access Denied', 
                'Page not found', 'Not Found', 'Blocked',
                'login', 'subscription', 'access required'
            ]
            content_lower = content.lower()
            
            # Small files (< 500 bytes) are likely error pages
            if file_size < 500:
                return ('error_page', 'Error page (too small)')
            
            # Check for error messages
            for indicator in error_indicators:
                if indicator.lower() in content_lower:
                    return ('error_page', f'Error page (contains: {indicator})')
            
            return ('html', 'HTML document')
        else:
            return ('unknown', 'Unknown format')
    except Exception as e:
        return ('unknown', f'Error checking: {str(e)}')

def main():
    script_dir = Path(__file__).parent
    json_path = script_dir / JSON_FILE
    output_dir = script_dir / OUTPUT_DIR
    report_path = script_dir / REPORT_FILE
    
    # Load JSON file
    print("Loading papers from JSON file...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    papers = data.get('papers', [])
    
    # Analyze downloads
    downloaded = []
    failed = []
    no_url = []
    missing_files = []
    
    for idx, paper in enumerate(papers, 1):
        title = paper.get('title', f'Paper_{idx}')
        url = paper.get('url')
        year = paper.get('year', '')
        authors = paper.get('authors', 'N/A')
        source = paper.get('source', 'N/A')
        
        filename = get_filename_for_paper(paper, idx)
        filepath = output_dir / filename
        
        paper_info = {
            'index': idx,
            'title': title,
            'authors': authors,
            'year': year,
            'source': source,
            'url': url,
            'filename': filename
        }
        
        if not url:
            no_url.append(paper_info)
        elif filepath.exists():
            file_type, file_desc = check_file_type(filepath)
            file_size = filepath.stat().st_size
            
            if file_type == 'placeholder':
                failed.append({**paper_info, 'reason': 'Download failed - placeholder file created'})
            elif file_type == 'invalid':
                failed.append({**paper_info, 'reason': 'Invalid file format'})
            elif file_type == 'error_page':
                failed.append({**paper_info, 'reason': f'Download failed - {file_desc}'})
            else:
                downloaded.append({**paper_info, 'file_size': file_size, 'file_type': file_type, 'file_desc': file_desc})
        else:
            missing_files.append({**paper_info, 'reason': 'File not found'})
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PAPER DOWNLOAD REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total papers in database: {len(papers)}")
    report_lines.append("")
    
    # Summary
    report_lines.append("SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Successfully downloaded: {len(downloaded)}")
    report_lines.append(f"Failed downloads: {len(failed)}")
    report_lines.append(f"No URL provided: {len(no_url)}")
    report_lines.append(f"Missing files (unexpected): {len(missing_files)}")
    report_lines.append("")
    
    # Downloaded papers
    report_lines.append("=" * 80)
    report_lines.append("SUCCESSFULLY DOWNLOADED PAPERS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total: {len(downloaded)}")
    report_lines.append("")
    
    for paper in sorted(downloaded, key=lambda x: x.get('year', 0) or 0, reverse=True):
        report_lines.append(f"[{paper['index']}] {paper['title']}")
        report_lines.append(f"     Authors: {paper['authors']}")
        report_lines.append(f"     Year: {paper['year']}")
        report_lines.append(f"     Source: {paper['source']}")
        file_desc = paper.get('file_desc', paper.get('file_type', 'unknown'))
        report_lines.append(f"     File: {paper['filename']} ({paper['file_size']:,} bytes, {file_desc})")
        report_lines.append(f"     URL: {paper['url']}")
        report_lines.append("")
    
    # Failed downloads
    report_lines.append("=" * 80)
    report_lines.append("FAILED DOWNLOADS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total: {len(failed)}")
    report_lines.append("")
    
    # Group failures by reason
    failures_by_reason = {}
    for paper in failed:
        reason = paper.get('reason', 'Unknown reason')
        if reason not in failures_by_reason:
            failures_by_reason[reason] = []
        failures_by_reason[reason].append(paper)
    
    for reason, papers_list in failures_by_reason.items():
        report_lines.append(f"\nReason: {reason}")
        report_lines.append("-" * 80)
        for paper in sorted(papers_list, key=lambda x: x.get('year', 0) or 0, reverse=True):
            report_lines.append(f"[{paper['index']}] {paper['title']}")
            report_lines.append(f"     Authors: {paper['authors']}")
            report_lines.append(f"     Year: {paper['year']}")
            report_lines.append(f"     Source: {paper['source']}")
            report_lines.append(f"     URL: {paper['url']}")
            report_lines.append("")
    
    # No URL
    report_lines.append("=" * 80)
    report_lines.append("PAPERS WITH NO URL")
    report_lines.append("=" * 80)
    report_lines.append(f"Total: {len(no_url)}")
    report_lines.append("")
    
    for paper in sorted(no_url, key=lambda x: x.get('year', 0) or 0, reverse=True):
        report_lines.append(f"[{paper['index']}] {paper['title']}")
        report_lines.append(f"     Authors: {paper['authors']}")
        report_lines.append(f"     Year: {paper['year']}")
        report_lines.append(f"     Source: {paper['source']}")
        report_lines.append("")
    
    # Missing files (shouldn't happen, but just in case)
    if missing_files:
        report_lines.append("=" * 80)
        report_lines.append("MISSING FILES (UNEXPECTED)")
        report_lines.append("=" * 80)
        report_lines.append(f"Total: {len(missing_files)}")
        report_lines.append("")
        
        for paper in missing_files:
            report_lines.append(f"[{paper['index']}] {paper['title']}")
            report_lines.append(f"     URL: {paper['url']}")
            report_lines.append(f"     Expected file: {paper['filename']}")
            report_lines.append("")
    
    # Detailed statistics by source
    report_lines.append("=" * 80)
    report_lines.append("STATISTICS BY SOURCE")
    report_lines.append("=" * 80)
    
    source_stats = {}
    for paper in papers:
        source = paper.get('source') or 'Unknown'
        if source not in source_stats:
            source_stats[source] = {'total': 0, 'downloaded': 0, 'failed': 0, 'no_url': 0}
        source_stats[source]['total'] += 1
    
    for paper in downloaded:
        source = paper.get('source') or 'Unknown'
        if source in source_stats:
            source_stats[source]['downloaded'] += 1
    
    for paper in failed:
        source = paper.get('source') or 'Unknown'
        if source in source_stats:
            source_stats[source]['failed'] += 1
    
    for paper in no_url:
        source = paper.get('source') or 'Unknown'
        if source in source_stats:
            source_stats[source]['no_url'] += 1
    
    # Sort sources, handling None values
    sorted_sources = sorted(source_stats.items(), key=lambda x: (x[0] is None, str(x[0]) if x[0] else ''))
    
    for source, stats in sorted_sources:
        if stats['total'] > 0:
            success_rate = (stats['downloaded'] / stats['total']) * 100 if stats['total'] > 0 else 0
            source_name = source if source else 'Unknown/None'
            report_lines.append(f"{source_name}:")
            report_lines.append(f"  Total: {stats['total']}")
            report_lines.append(f"  Downloaded: {stats['downloaded']} ({success_rate:.1f}%)")
            report_lines.append(f"  Failed: {stats['failed']}")
            report_lines.append(f"  No URL: {stats['no_url']}")
            report_lines.append("")
    
    # Write report
    report_content = "\n".join(report_lines)
    
    # Write to file
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\nReport generated successfully!")
    print(f"Report saved to: {report_path.absolute()}")
    print(f"\nSummary:")
    print(f"  Downloaded: {len(downloaded)}")
    print(f"  Failed: {len(failed)}")
    print(f"  No URL: {len(no_url)}")

if __name__ == "__main__":
    main()


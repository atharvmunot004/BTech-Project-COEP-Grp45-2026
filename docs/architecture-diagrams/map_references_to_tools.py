import json
import os
from pathlib import Path
from difflib import SequenceMatcher
import re

def normalize_title(title):
    """Normalize title for comparison"""
    if not title:
        return ""
    # Remove extra whitespace, convert to lowercase
    title = re.sub(r'\s+', ' ', title.strip().lower())
    # Remove special characters but keep spaces
    title = re.sub(r'[^\w\s]', '', title)
    return title

def normalize_authors(authors):
    """Normalize authors for comparison"""
    if not authors:
        return ""
    # Remove extra whitespace, convert to lowercase
    authors = re.sub(r'\s+', ' ', str(authors).strip().lower())
    # Remove common prefixes and suffixes
    authors = re.sub(r'\bet al\.?', '', authors)
    return authors.strip()

def similarity(str1, str2):
    """Calculate similarity ratio between two strings"""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1, str2).ratio()

def match_reference(tool_ref, paper):
    """Check if a tool reference matches a paper"""
    score = 0.0
    matches = []
    
    # URL matching (most reliable, check first)
    tool_url = (tool_ref.get('url') or '').strip()
    paper_url = (paper.get('url') or '').strip()
    if tool_url and paper_url:
        # Normalize URLs (remove trailing slashes, convert to lowercase for comparison)
        tool_url_norm = tool_url.rstrip('/').lower()
        paper_url_norm = paper_url.rstrip('/').lower()
        if tool_url_norm == paper_url_norm:
            score += 1.0  # Perfect match if URLs match
            matches.append("url")
            return score, matches  # URL match is definitive
    
    # Title matching (important)
    tool_title = normalize_title(tool_ref.get('title', ''))
    paper_title = normalize_title(paper.get('title', ''))
    if tool_title and paper_title:
        title_sim = similarity(tool_title, paper_title)
        if title_sim > 0.85:  # High threshold for title match
            score += title_sim * 0.5
            matches.append(f"title:{title_sim:.2f}")
    
    # Author matching
    tool_authors = normalize_authors(tool_ref.get('authors', ''))
    paper_authors = normalize_authors(paper.get('authors', ''))
    if tool_authors and paper_authors:
        author_sim = similarity(tool_authors, paper_authors)
        if author_sim > 0.7:
            score += author_sim * 0.3
            matches.append(f"authors:{author_sim:.2f}")
    
    # Year matching
    tool_year = tool_ref.get('year')
    paper_year = paper.get('year')
    if tool_year and paper_year and tool_year == paper_year:
        score += 0.15
        matches.append("year")
    
    return score, matches

def extract_tool_name_from_filename(filename):
    """Extract tool name from filename"""
    # Remove .json or .md extension and number prefix
    name = filename.replace('.json', '').replace('.md', '')
    # Remove leading number and dash (e.g., "001-")
    name = re.sub(r'^\d+-', '', name)
    return name

def parse_markdown_references(md_file_path):
    """Parse references from markdown proposal files"""
    references = []
    
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the Literature section (could be "Literature & Motivation" or "Literature & References")
        lit_pattern = r'##\s*2\.\s*Literature[^\n]*\n(.*?)(?=\n##|\Z)'
        lit_match = re.search(lit_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if not lit_match:
            return references
        
        lit_section = lit_match.group(1)
        
        # Extract link definitions at the bottom (format: [1]: https://url)
        link_pattern = r'\[(\d+)\]:\s*(https?://[^\s]+)'
        links = {}
        for match in re.finditer(link_pattern, content):
            links[match.group(1)] = match.group(2)
        
        # Extract reference lines (format: - Author (Year) – Description. ([Source][link_number]))
        # The description might be a short description, not the full title, so we'll use URL for matching
        # Pattern: - Author (Year) – Description. ([Source][N])
        ref_pattern = r'-\s*([^(]+?)\s*\((\d{4})\)\s*[–-]\s*([^.]+?)\.\s*\(\[([^\]]+)\]\[(\d+)\]\)'
        for match in re.finditer(ref_pattern, lit_section):
            authors = match.group(1).strip()
            year = int(match.group(2))
            description = match.group(3).strip()  # This might be a description, not full title
            source = match.group(4).strip()
            link_num = match.group(5)
            url = links.get(link_num, '')
            
            ref = {
                'authors': authors,
                'year': year,
                'title': description,  # Use description as title hint, but URL is more reliable
                'source': source,
                'url': url
            }
            references.append(ref)
        
    except Exception as e:
        print(f"  Error parsing {md_file_path}: {e}")
    
    return references

def main():
    # Paths
    llm_jsons_dir = Path(__file__).parent.parent / 'llm-jsons'
    proposal_dir = Path(__file__).parent.parent / 'proposal'
    references_file = Path(__file__).parent / 'references_list_details.json'
    
    # Load references
    with open(references_file, 'r', encoding='utf-8') as f:
        references_data = json.load(f)
    
    papers = references_data['papers']
    
    # Create a mapping: paper -> list of tools
    paper_to_tools = {i: [] for i in range(len(papers))}
    
    # Read all tool JSON files
    tool_files = sorted(llm_jsons_dir.glob('*.json'))
    
    print(f"Processing {len(tool_files)} tool JSON files...")
    
    for tool_file in tool_files:
        tool_name = extract_tool_name_from_filename(tool_file.name)
        print(f"\nProcessing tool: {tool_name}")
        
        with open(tool_file, 'r', encoding='utf-8') as f:
            tool_data = json.load(f)
        
        # Extract references from "references" section
        tool_refs = tool_data.get('references', [])
        
        # Also check extensions_research_directions for references
        extensions = tool_data.get('extensions_research_directions', [])
        for ext in extensions:
            ext_refs = ext.get('references', [])
            if isinstance(ext_refs, list):
                for ref in ext_refs:
                    if isinstance(ref, str):
                        # Try to parse string references
                        tool_refs.append({'title': ref})
        
        # Match each tool reference to papers
        for tool_ref in tool_refs:
            best_match_idx = None
            best_score = 0.0
            best_matches = []
            
            for i, paper in enumerate(papers):
                score, matches = match_reference(tool_ref, paper)
                if score > best_score:
                    best_score = score
                    best_match_idx = i
                    best_matches = matches
            
            # If we found a good match, add tool to paper's tools list
            if best_match_idx is not None and best_score > 0.6:
                if tool_name not in paper_to_tools[best_match_idx]:
                    paper_to_tools[best_match_idx].append(tool_name)
                    paper_title = papers[best_match_idx].get('title', 'Unknown')
                    print(f"  Matched: {tool_ref.get('title', 'Unknown')[:50]}... -> {paper_title[:50]}... (score: {best_score:.2f}, {', '.join(best_matches)})")
            elif tool_ref.get('title'):
                print(f"  No match found for: {tool_ref.get('title', 'Unknown')[:50]}...")
    
    # Process proposal markdown files
    proposal_files = sorted(proposal_dir.glob('*.md'))
    # Filter out non-tool files (like proposal.md, research-prospects.md, etc.)
    proposal_files = [f for f in proposal_files if re.match(r'^\d+-', f.name)]
    
    print(f"\n\nProcessing {len(proposal_files)} proposal markdown files...")
    
    for proposal_file in proposal_files:
        tool_name = extract_tool_name_from_filename(proposal_file.name)
        print(f"\nProcessing proposal: {tool_name}")
        
        # Parse references from markdown
        tool_refs = parse_markdown_references(proposal_file)
        
        # Match each reference to papers
        for tool_ref in tool_refs:
            best_match_idx = None
            best_score = 0.0
            best_matches = []
            
            for i, paper in enumerate(papers):
                score, matches = match_reference(tool_ref, paper)
                if score > best_score:
                    best_score = score
                    best_match_idx = i
                    best_matches = matches
            
            # If we found a good match, add tool to paper's tools list
            if best_match_idx is not None and best_score > 0.6:
                if tool_name not in paper_to_tools[best_match_idx]:
                    paper_to_tools[best_match_idx].append(tool_name)
                    paper_title = papers[best_match_idx].get('title', 'Unknown')
                    print(f"  Matched: {tool_ref.get('title', 'Unknown')[:50]}... -> {paper_title[:50]}... (score: {best_score:.2f}, {', '.join(best_matches)})")
            elif tool_ref.get('title'):
                print(f"  No match found for: {tool_ref.get('title', 'Unknown')[:50]}...")
    
    # Update references with tools
    print(f"\n\nUpdating references with tools...")
    for i, paper in enumerate(papers):
        tools = paper_to_tools[i]
        paper['tools-used-in'] = sorted(tools) if tools else []
        if tools:
            print(f"  {paper.get('title', 'Unknown')[:60]}... -> {len(tools)} tool(s): {', '.join(tools)}")
    
    # Save updated references
    with open(references_file, 'w', encoding='utf-8') as f:
        json.dump(references_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nDone! Updated {references_file}")
    print(f"Total papers: {len(papers)}")
    papers_with_tools = sum(1 for p in papers if p['tools-used-in'])
    print(f"Papers with tools: {papers_with_tools}")

if __name__ == '__main__':
    main()


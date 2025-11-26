import json

# Read the Excalidraw file
with open('architecture_diagram.excalidraw', 'r', encoding='utf-8') as f:
    data = json.load(f)

orange_tools = []
green_tools = []

elements = data.get('elements', [])

# Extract orange tools (#f08c00)
for el in elements:
    if el.get('strokeColor') == '#f08c00' and el.get('type') == 'text' and el.get('text'):
        text = el.get('text', '').replace('\n', ' ').strip()
        if text:  # Only add if text is not empty
            orange_tools.append({
                'id': el.get('id'),
                'name': text,
                'type': 'text'
            })

# Extract green tools (#2f9e44)
for el in elements:
    if el.get('strokeColor') == '#2f9e44' and el.get('type') == 'text' and el.get('text'):
        text = el.get('text', '').replace('\n', ' ').strip()
        if text:  # Only add if text is not empty
            green_tools.append({
                'id': el.get('id'),
                'name': text,
                'type': 'text'
            })

# Create result structure
result = {
    'orange_tools': orange_tools,
    'green_tools': green_tools,
    'summary': {
        'total_orange_tools': len(orange_tools),
        'total_green_tools': len(green_tools),
        'total_tools': len(orange_tools) + len(green_tools)
    }
}

# Write to JSON file
with open('tools.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(orange_tools)} orange tools and {len(green_tools)} green tools")
print("Results saved to tools.json")


# Fix the incomplete JSON file
import json

filepath = r'c:\projects\inference\xinference\model\llm\llm_family.json'

# Read file content
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Find where the previous entry ends before our incomplete mineru entry
# Look for the pattern where our entry begins
mineru_idx = content.find('"model_name": "mineru-vlm"')

if mineru_idx > 0:
    # Find the start of our entry (the { before model_name)
    entry_start = content.rfind('{', 0, mineru_idx)
    
    # Get content before our entry (should end with },)
    good_content = content[:entry_start].rstrip()
    if good_content.endswith(','):
        good_content = good_content[:-1]  # Remove trailing comma
    
    # Close the JSON array properly
    good_content += '\n]'
    
    print(f"Original file size: {len(content)}")
    print(f"Fixed file size: {len(good_content)}")
    
    # Write fixed content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(good_content)
    
    print("JSON fixed - reverted to state before mineru-vlm entry")
    
    # Now verify the JSON is valid
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"JSON is valid with {len(data)} models")
else:
    print("mineru-vlm entry not found")

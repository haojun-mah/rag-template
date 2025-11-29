from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from unstructured.partition.auto import partition
from typing import List, Dict
import os

def parse_html(file_path: str) -> List[Dict]:
    """Parse a HTML file with unstructured library and return a list of information"""
    
    try: 
        # Use partition_auto which automatically detects the file type
        # For SEC filings with embedded HTML, this works better than partition_html
        elements = partition(
            filename=file_path,
            strategy='fast',  # Use hi_res for better accuracy with complex documents
            infer_table_structure=True,
            include_page_breaks=True,
            extract_images_in_pdf=False
        )
        return [el.to_dict() for el in elements]
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return []


# Test code below 
folder_path = "sec-edgar-filings/MSFT"
file_list = []

for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_list.append(os.path.join(root, file))

print(f"Found {len(file_list)} files")
print(file_list)

# Get a 10-K file
ten_k_file = [f for f in file_list if "10-K" in f][0]
print(f"\nParsing file: {ten_k_file}...")

parsed_elements = parse_html(ten_k_file)

print(f"\nSuccessfully parsed into {len(parsed_elements)} elements.")
print("\n--- Sample Elements ---")

# Print a few sample elements to inspect their type and content
for i, element in enumerate(parsed_elements[20:25]): # Show a slice of elements
    elem_type = element.get('type', 'N/A')
    text_snippet = element.get('text', '')[:100].replace('\n', ' ') + '...'
    print(f"Element {i+20}: [Type: {elem_type}] - Content: '{text_snippet}'")

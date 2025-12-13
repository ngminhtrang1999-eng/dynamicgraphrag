import os
import re

def natural_sort_key(s):
    """
    Key to sort strings containing numbers naturally.
    E.g., ensures 'Chapter 2' comes before 'Chapter 10'.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def merge_files_in_folder(folder_path, output_filename):
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Skipping: Folder '{folder_path}' not found.")
        return

    # Get all .txt files
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    # Sort files naturally (1, 2, ..., 10)
    files.sort(key=natural_sort_key)

    if not files:
        print(f"No txt files found in {folder_path}")
        return

    print(f"Processing folder: {folder_path}...")
    
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    
                    outfile.write(content)
                    
                    # Add a newline at the end of each file content
                    outfile.write("\n\n")
                    
                print(f"  -> Merged: {filename}")
            except Exception as e:
                print(f"  -> Error reading {filename}: {e}")

    print(f"Done! Saved to: {output_filename}\n")

# --- Configuration ---

# The exact folder names from your screenshot
folders_to_process = [
    "C:/Users/trang/OneDrive/Desktop/Dynamic GraphRAG/data/LOTM/translations/gemini-2.5-flash_No_RAG",
    "C:/Users/trang/OneDrive/Desktop/Dynamic GraphRAG/data/LOTM/translations/gemini-2.5-flash_RAG"
]

# Run the merge for each folder
for folder in folders_to_process:
    # Create an output name based on the folder name
    # e.g., "gemini-2.5-flash No RAG_FULL.txt"
    output_name = f"{folder}_FULL.txt"
    merge_files_in_folder(folder, output_name)
import os
import sys
import spacy
import re

nlp = spacy.load("en_core_web_trf")

def get_spacy_candidates(text):
    nlp.max_length = len(text) + 100000
    doc = nlp(text)
    names = set()
    for ent in doc.ents:
        print(f"Entity: {ent.text} | Label: {ent.label_}")
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT"]:
        # if ent.label_ in ["PERSON"]:
            clean = re.sub(r"['’]s\b", "", ent.text, flags=re.IGNORECASE)
            clean = re.sub(r"[^\w\s]", " ", clean)
            clean = re.sub(r"\s+", " ", clean).strip()
            if len(clean) > 2 and not clean.isdigit():
                names.add(clean)
    return list(names)

def main():
    # Default chapter path
    default_chapter_path = r"d:\Dynamic GraphRAG\data\LOTM\chapters\chapter_0001.txt"
    
    # Check if a file path is provided as a command line argument
    if len(sys.argv) > 1:
        chapter_path = sys.argv[1]
    else:
        chapter_path = default_chapter_path
        print(f"No file provided, using default: {chapter_path}")

    if not os.path.exists(chapter_path):
        print(f"Error: File not found at {chapter_path}")
        return

    try:
        with open(chapter_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        print(f"--- Processing: {os.path.basename(chapter_path)} ---")
        print(f"Text length: {len(text)} characters")
        
        entities = get_spacy_candidates(text)
        
        print("\n--- Extracted Entities ---")
        for i, entity in enumerate(entities, 1):
            print(f"{i}. {entity}")
            
        print(f"\nTotal entities found: {len(entities)}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

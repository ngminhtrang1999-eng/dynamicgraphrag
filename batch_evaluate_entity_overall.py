import pandas as pd
import os
from utils import preprocess_evaluate_data, get_spacy_candidates

# Configuration
data_dir = "evaluation/alignment_results"
output_dir = "evaluation"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

models_to_eval = ["RAG", "No_RAG", "GG"]

print("Starting OVERALL Entity evaluation...")

# Dictionary to hold aggregated data for each model
aggregated_data = {model: [] for model in models_to_eval}

# 1. Load and Preprocess Data (Chapters 1-10)
for chapter in range(1, 11):
    filename = f"aligned_master_chap_{chapter}.csv"
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Collecting data from {filename}...")
        try:
            df = pd.read_csv(filepath)
            
            if "ref" not in df.columns:
                print(f"SKIP {chapter}: Column 'ref' not found.")
                continue

            for model_name in models_to_eval:
                if model_name in df.columns:
                    raw_rows = []
                    for _, row in df.iterrows():
                        raw_rows.append({
                            "src": row.get("Source_Text", ""),
                            "mt": row[model_name],
                            "ref": row["ref"]
                        })
                    
                    # Preprocess and merge logic (handles multi-line merging)
                    clean_rows = preprocess_evaluate_data(raw_rows)
                    if clean_rows:
                        aggregated_data[model_name].extend(clean_rows)
                else:
                    print(f"  Column '{model_name}' missing in {filename}.")
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    else:
        print(f"File not found: {filename}")

# 2. Define Entity Evaluation Function
def evaluate_entity_accuracy(data_list, model_name="Model"):
    """
    Calculates Entity Accuracy (M-ETA style):
    Extracts entities from Ref, checks if they exist in MT.
    Score = Correct / Total
    """
    total_entities = 0
    correct_entities = 0
    
    print(f"Evaluating entities for {model_name} ({len(data_list)} samples)...")
    
    for i, item in enumerate(data_list):
        ref = str(item['ref'])
        mt = str(item['mt'])
        
        if not ref.strip():
            continue
            
        # Extract entities from Reference (acting as Ground Truth)
        ref_entities = get_spacy_candidates(ref)
        
        if not ref_entities:
            continue
            
        total_entities += len(ref_entities)
        
        # Check matches in MT
        mt_lower = mt.lower()
        for ent in ref_entities:
            # Case-insensitive substring match
            if ent.lower() in mt_lower:
                correct_entities += 1
        
        # Optional: Progress logging
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} samples...")
            
    accuracy = correct_entities / total_entities if total_entities > 0 else 0.0
    return accuracy, correct_entities, total_entities

# 3. Calculate Overall Scores
results = []
score_row = {"Type": "Entity Accuracy"}
details_row = {"Type": "Details (Correct/Total)"}

for model_name in models_to_eval:
    data_list = aggregated_data[model_name]
    if data_list:
        accuracy, correct, total = evaluate_entity_accuracy(data_list, model_name)
        score_row[model_name] = accuracy
        details_row[model_name] = f"{correct}/{total}"
    else:
        score_row[model_name] = None
        details_row[model_name] = None

results.append(score_row)
# results.append(details_row) # Optional: uncomment if you want raw counts in the CSV

results_df = pd.DataFrame(results)

if not results_df.empty:
    # Identify Best Model
    valid_cols = [c for c in models_to_eval if c in results_df.columns and pd.notna(results_df[c].iloc[0])]
    if valid_cols:
        results_df["Best"] = results_df[valid_cols].idxmax(axis=1)
    
    output_path = os.path.join(output_dir, "entity_overall_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nFinal Overall Results saved to {output_path}")
    print(results_df)
else:
    print("No results generated.")

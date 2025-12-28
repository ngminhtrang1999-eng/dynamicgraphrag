from bleurt import score
import pandas as pd
import os
from utils import preprocess_evaluate_data

# Bạn cần tải checkpoint BLEURT về máy trước (ví dụ: BLEURT-20)
# Link tải thường có trên GitHub của Google Research
checkpoint = "BLEURT-20" 

try:
    print(f"Loading BLEURT scorer from {checkpoint}...")
    scorer = score.BleurtScorer(checkpoint)
except Exception as e:
    print(f"Error loading BLEURT checkpoint: {e}")
    exit()

data_dir = "evaluation/alignment_results"
output_dir = "evaluation"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

models_to_eval = ["RAG", "No_RAG", "GG"]

print("Starting OVERALL evaluation for BLEURT...")

# Dictionary to hold aggregated data for each model
aggregated_data = {model: [] for model in models_to_eval}

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
                    
                    # Preprocess and merge logic
                    clean_rows = preprocess_evaluate_data(raw_rows)
                    if clean_rows:
                        aggregated_data[model_name].extend(clean_rows)
                else:
                    print(f"  Column '{model_name}' missing in {filename}.")
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    else:
        print(f"File not found: {filename}")

# Calculate Overall Score
results = []
score_row = {"Type": "Overall Score"}

for model_name in models_to_eval:
    data_list = aggregated_data[model_name]
    if data_list:
        print(f"Calculating overall score for {model_name} with {len(data_list)} samples...")
        refs = [r['ref'] for r in data_list]
        cands = [r['mt'] for r in data_list]
        
        scores = scorer.score(references=refs, candidates=cands)
        avg_score = sum(scores) / len(scores) if scores else 0
        score_row[model_name] = avg_score
    else:
        score_row[model_name] = None

results.append(score_row)
results_df = pd.DataFrame(results)

if not results_df.empty:
    valid_cols = [c for c in models_to_eval if c in results_df.columns]
    if valid_cols:
        results_df["Best"] = results_df[valid_cols].idxmax(axis=1)
    
    output_path = os.path.join(output_dir, "bleurt-20_overall_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nFinal Overall Results saved to {output_path}")
    print(results_df)
else:
    print("No results generated.")

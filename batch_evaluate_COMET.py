from comet import download_model, load_from_checkpoint
import pandas as pd
import os
import torch
from utils import preprocess_evaluate_data

# 1. Tải model COMET
# Model tiêu chuẩn: wmt22-comet-da
model_id = "Unbabel/wmt22-comet-da"
print(f"Downloading/Loading COMET model: {model_id}...")
model_path = download_model(model_id)
model = load_from_checkpoint(model_path)

# Check GPU
gpus = 1 if torch.cuda.is_available() else 0
print(f"Using GPUs: {gpus}")

data_dir = "evaluation/alignment_results"
output_dir = "evaluation"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

models_to_eval = ["RAG", "No_RAG", "GG"]

print("Starting OVERALL evaluation for COMET...")

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
        
        # COMET expects list of dicts: {"src": ..., "mt": ..., "ref": ...}
        # data_list already in that format
        
        model_output = model.predict(data_list, batch_size=8, gpus=gpus)
        score_row[model_name] = model_output.system_score
    else:
        score_row[model_name] = None

results.append(score_row)
results_df = pd.DataFrame(results)

if not results_df.empty:
    valid_cols = [c for c in models_to_eval if c in results_df.columns]
    if valid_cols:
        results_df["Best"] = results_df[valid_cols].idxmax(axis=1)
    
    output_path = os.path.join(output_dir, "comet_overall_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nFinal Overall Results saved to {output_path}")
    print(results_df)
else:
    print("No results generated.")

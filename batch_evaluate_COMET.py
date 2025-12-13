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


results = []
models_to_eval = ["RAG", "No_RAG", "GG"]

print("Starting COMET evaluation...")

for chapter in range(1, 11):
    chapter_scores = {"Chapter": f"Chapter {chapter}"}
    
    # New filename format
    filename = f"aligned_master_chap_{chapter}.csv"
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Processing {filename}...")
        try:
            df = pd.read_csv(filepath)
            
            # Check required columns
            # Column 'ref' is the anchor. 'RAG', 'No_RAG', 'GG' are candidates.
            if "ref" not in df.columns:
                print(f"SKIP {chapter}: Column 'ref' not found.")
                continue

            for model_name in models_to_eval:
                if model_name in df.columns:
                    # 1. Prepare raw list
                    raw_rows = []
                    for _, row in df.iterrows():
                        raw_rows.append({
                            "src": row.get("Source_Text", ""),
                            "mt": row[model_name],
                            "ref": row["ref"]
                        })
                    
                    # 2. Preprocess (Merge logic)
                    clean_rows = preprocess_evaluate_data(raw_rows)
                    
                    if not clean_rows:
                        print(f"  Warning: No valid data for {model_name} in Chap {chapter}")
                        chapter_scores[model_name] = 0
                        continue

                    # 3. Prepare data for COMET (list of dicts with src, mt, ref)
                    # clean_rows is alrady in the format [{"src":..., "mt":..., "ref":...}]
                    comet_data = clean_rows
                    
                    # 4. Score
                    # batch_size=8, gpus based on check
                    model_output = model.predict(comet_data, batch_size=8, gpus=gpus)
                    # model_output.system_score is the average
                    
                    chapter_scores[model_name] = model_output.system_score
                    # print(f"  {model_name}: {model_output.system_score:.4f}")
                else:
                    print(f"  Column '{model_name}' missing in CSV.")
                    chapter_scores[model_name] = None
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    else:
        print(f"File not found: {filename}")
        
    results.append(chapter_scores)

results_df = pd.DataFrame(results)

# Post-processing
if not results_df.empty:
    # Identify best
    # Filter columns that actually exist and are numeric
    valid_cols = [c for c in models_to_eval if c in results_df.columns]
    if valid_cols:
        results_df["Best"] = results_df[valid_cols].idxmax(axis=1)
    
    # Reorder
    cols = ["Chapter"] + valid_cols + (["Best"] if valid_cols else [])
    results_df = results_df[cols]

    output_path = os.path.join(output_dir, "comet_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nFinal Results saved to {output_path}")
    print(results_df)
else:
    print("No results generated.")

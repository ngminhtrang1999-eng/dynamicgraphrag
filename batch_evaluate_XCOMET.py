from comet import download_model, load_from_checkpoint
import pandas as pd
import os
import torch
import json
from utils import preprocess_evaluate_data, save_json

# 1. Load XCOMET model
# Model: Unbabel/XCOMET-XXL
model_id = "Unbabel/XCOMET-XL"
print(f"Downloading/Loading XCOMET model: {model_id}...")
try:
    model_path = download_model(model_id)
    model = load_from_checkpoint(model_path)
except Exception as e:
    print(f"Error loading model {model_id}: {e}")
    print("Please ensure you have access to this gated model on HuggingFace and are logged in (huggingface-cli login).")
    exit(1)

# Check GPU
gpus = 1 if torch.cuda.is_available() else 0
print(f"Using GPUs: {gpus}")

data_dir = "evaluation/alignment_results_llama-4-maverick-17b-128e"
output_dir = "evaluation"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

models_to_eval = ["RAG", "No_RAG", "GG"]

print("Starting OVERALL evaluation for XCOMET...")

# Dictionary to hold aggregated data for each model
aggregated_data = {model: [] for model in models_to_eval}
# Also keep track of source attribution for details (Chapter, Row Index) if needed, 
# but for overall aggregation we usually just lump them. 
# To make details useful, we might want to structure them by Chapter in the details JSON, 
# but the prompt implies a batch run. Let's store sufficient metadata if possible.
# Current `aggregated_data` is just a list of dicts. We can add metadata to these dicts, 
# but COMET predict expects specific keys. However, we can pass extra keys and COMET might ignore them,
# or we just rely on index alignment. 
# Better: Make `aggregated_data` a list of dicts, where each dict has "src", "mt", "ref", "meta": {...}
# But `model.predict` acts on a list of dicts. 
# XCOMET `predict` returns an object with `scores`, `system_score`, `metadata.error_spans`.

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
                    for idx, row in df.iterrows():
                        raw_rows.append({
                            "src": row.get("Source_Text", ""),
                            "mt": row[model_name],
                            "ref": row["ref"],
                            # Internal tracking, utils.preprocess might strip this if not careful
                            # strict utils.preprocess_evaluate_data only keeps src, mt, ref if it reconstructs new dicts.
                            # Let's check utils.py... it rebuilds dicts. So metadata is lost.
                            # We will rely on simple aggregation.
                        })
                    
                    # Preprocess and merge logic
                    clean_rows = preprocess_evaluate_data(raw_rows)
                    if clean_rows:
                        # Append to model's list
                        aggregated_data[model_name].extend(clean_rows)
                else:
                    print(f"  Column '{model_name}' missing in {filename}.")
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    else:
        print(f"File not found: {filename}")

# Calculate Overall Score and Details
results = []
score_row = {"Type": "Overall Score"}
detailed_outputs = {}

for model_name in models_to_eval:
    data_list = aggregated_data[model_name]
    if data_list:
        print(f"Calculating XCOMET for {model_name} with {len(data_list)} samples...")
        
        # data_list has {"src", "mt", "ref"}
        try:
            model_output = model.predict(data_list, batch_size=8, gpus=gpus)
            
            # System score
            score_row[model_name] = model_output.system_score
            
            # Detailed outputs
            # model_output.scores: segment level scores
            # model_output.metadata.error_spans: error spans
            
            detailed_outputs[model_name] = {
                "system_score": model_output.system_score,
                "segment_scores": model_output.scores,
                "error_spans": model_output.metadata.error_spans
            }
            
        except Exception as e:
            print(f"Error predicting for {model_name}: {e}")
            score_row[model_name] = None
    else:
        score_row[model_name] = None

results.append(score_row)
results_df = pd.DataFrame(results)

if not results_df.empty:
    valid_cols = [c for c in models_to_eval if c in results_df.columns]
    if valid_cols:
        results_df["Best"] = results_df[valid_cols].idxmax(axis=1)
    
    # Save CSV
    output_path_csv = os.path.join(output_dir, "xcomet_overall_results.csv")
    results_df.to_csv(output_path_csv, index=False)
    print(f"\nFinal Overall Results saved to {output_path_csv}")
    print(results_df)

    # Save Details JSON
    # Convert numpy types to python types for JSON serialization if needed
    # error_spans usually list of lists of objects.
    # Scores are usually list of floats.
    output_path_json = os.path.join(output_dir, "xcomet_details.json")
    
    # Helper to clean for JSON
    def clean_for_json(obj):
        if isinstance(obj, (float, int)):
            return obj
        if hasattr(obj, "tolist"): # Numpy arrays
            return obj.tolist()
        if isinstance(obj, list):
            return [clean_for_json(x) for x in obj]
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        return obj

    try: 
        cleaned_details = clean_for_json(detailed_outputs)
        save_json(output_path_json, cleaned_details)
        print(f"Detailed results (error spans) saved to {output_path_json}")
    except Exception as e:
        print(f"Error saving details JSON: {e}")

else:
    print("No results generated.")

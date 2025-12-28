import os
import json
import re
import argparse
from google import genai
from google.genai import types

import const
from utils import split_text_with_overlap
from modules.prompts import build_translation_prompt

# Ensure output dir exists
BATCH_DIR = os.path.join(const.DATA_DIR, "batch_processing")
os.makedirs(BATCH_DIR, exist_ok=True)

INPUT_JSONL = os.path.join(BATCH_DIR, "batch_input.jsonl")
OUTPUT_JSONL = os.path.join(BATCH_DIR, "batch_output.jsonl")

def convert_messages_to_gemini_request(messages, model_config=None):
    """
    Converts OpenAI-style messages (list of dicts) into a native Gemini 
    GenerateContentRequest dictionary.
    """
    contents = []
    system_instruction = None

    for msg in messages:
        role = msg["role"]
        content_text = msg["content"]
        
        if role == "system":
            # Native Gemini supports system_instruction as a separate field
            system_instruction = {"parts": [{"text": content_text}]}
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content_text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content_text}]})

    # Build the request object
    request_body = {
        "contents": contents,
        "generationConfig": {
            "temperature": const.TEMP_CREATIVE,
            "maxOutputTokens": const.MAX_NEW_TOKENS,
        }
    }

    if system_instruction:
        request_body["system_instruction"] = system_instruction

    return request_body

def step_1_prepare_batch_file():
    print(f"🚀 STEP 1: Preparing Native Batch File for Chapters {const.START_CHAPTER} to {const.END_CHAPTER}...")
    
    all_files = sorted([f for f in os.listdir(const.INPUT_DIR) if f.endswith(".txt")])
    request_buffer = []
    total_chunks = 0
    
    for chap_idx in range(const.START_CHAPTER, const.END_CHAPTER + 1):
        file_name = next((f for f in all_files if str(chap_idx) in f), None)
        if not file_name:
            continue
            
        with open(os.path.join(const.INPUT_DIR, file_name), "r", encoding="utf-8") as f:
            text = re.sub(r'\s+', ' ', f.read())
        
        chunks = split_text_with_overlap(text, const.CHUNK_SIZE, 0)
        
        for chunk_idx, chunk in enumerate(chunks):
            # 1. Get prompts (OpenAI style)
            messages = build_translation_prompt(chunk)
            
            # 2. Convert to Gemini Native Request Format
            gemini_request = convert_messages_to_gemini_request(messages)
            
            # 3. Create Batch Item (Native Format: 'key' + 'request')
            batch_item = {
                "key": f"chap-{chap_idx}-chunk-{chunk_idx}",
                "request": gemini_request
            }
            
            request_buffer.append(json.dumps(batch_item))
            total_chunks += 1
            
        print(f"   + Added Chapter {chap_idx} ({len(chunks)} chunks)")

    with open(INPUT_JSONL, "w", encoding="utf-8") as f:
        f.write("\n".join(request_buffer))
        
    print(f"✅ Created {INPUT_JSONL} with {total_chunks} requests.")

def step_2_submit_batch():
    print("🚀 STEP 2: Uploading and Submitting Batch Job...")
    client = genai.Client(api_key=const.GEMINI_API_KEY)
    
    # 1. Upload File with correct MIME type for JSONL
    print(f"   Uploading {INPUT_JSONL}...")
    batch_file = client.files.upload(
        file=INPUT_JSONL,
        config={'mime_type': 'application/json'} 
    )
    print(f"   File Uploaded: {batch_file.name}")
    
    # 2. Submit Job
    print(f"   Submitting job for model: {const.GEMINI_MODEL_NAME}...")
    batch_job = client.batches.create(
        model=const.GEMINI_MODEL_NAME,
        src=batch_file.name,
        config=types.CreateBatchJobConfig(
            display_name=f"trans_chap_{const.START_CHAPTER}_{const.END_CHAPTER}"
        )
    )
    
    print(f"✅ Job Started! Job Name: {batch_job.name}")
    print(f"   State: {batch_job.state}")
    print("   NOTE: Save this Job Name to check status later.")
    return batch_job.name

def step_3_check_and_download(job_name):
    print(f"🚀 STEP 3: Checking Job {job_name}...")
    client = genai.Client(api_key=const.GEMINI_API_KEY)
    
    try:
        batch_job = client.batches.get(name=job_name)
    except Exception as e:
        print(f"❌ Error fetching job: {e}")
        return False
    
    # Handle State Enum safely
    current_state = batch_job.state
    state_name = current_state.name if hasattr(current_state, 'name') else str(current_state)
    
    print(f"   Current State: {state_name}")
    
    if state_name == 'JOB_STATE_SUCCEEDED':
        print("   Job Finished! Downloading results...")
        
        # 1. Determine the output file name
        result_file_name = None
        if hasattr(batch_job, 'output_file') and batch_job.output_file:
             result_file_name = batch_job.output_file
        elif hasattr(batch_job, 'dest') and batch_job.dest and hasattr(batch_job.dest, 'file_name'):
             result_file_name = batch_job.dest.file_name

        if result_file_name:
            print(f"   Target File: {result_file_name}")
            try:
                # 2. Use the Native SDK Download method
                # FIX: Changed 'name=' to 'file='
                content = client.files.download(file=result_file_name)
                
                with open(OUTPUT_JSONL, "wb") as f:
                    f.write(content)
                
                print(f"✅ Results saved to {OUTPUT_JSONL}")
                return True
                
            except Exception as e:
                print(f"❌ SDK Download Failed: {e}")
                # Known Issue Check: If this fails with a 400 'File ID too long' error, 
                # it is a known bug on the Google backend for Batch jobs.
                if "400" in str(e) and "File ID" in str(e):
                    print("   ⚠️ KNOWN ISSUE: The Batch ID generated by Google is too long for the Files API.")
                    print("   Please try deleting this job and submitting a new one, or wait for a Google fix.")
                return False
        else:
            print("❌ Job succeeded but could not determine output file name.")
            return False

    elif state_name == 'JOB_STATE_FAILED':
        print(f"❌ Job Failed. Error: {batch_job.error}")
        return False
        
    else:
        print(f"   Job is still processing (State: {state_name}). Please check again later.")
        return False
    
def step_4_reassemble_files():
    print("🚀 STEP 4: Reassembling Chapters (Native Format)...")
    
    if not os.path.exists(OUTPUT_JSONL):
        print("❌ Output file not found. Run Step 3 first.")
        return

    results_map = {} 
    
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            
            try:
                item = json.loads(line)
                custom_id = item.get("key")
                if not custom_id:
                    continue

                parts = custom_id.split("-")
                chap_id = parts[1]
                chunk_id = int(parts[3])
                
                if "response" in item:
                    candidates = item["response"].get("candidates", [])
                    if candidates and candidates[0].get("content"):
                        trans_text = candidates[0]["content"]["parts"][0]["text"]
                    else:
                        trans_text = "[NO CONTENT RETURNED]"
                else:
                    error_msg = item.get("status", {}).get("message", "Unknown Error")
                    print(f"   ⚠️ API Error in {custom_id}: {error_msg}")
                    trans_text = f"[API ERROR: {error_msg}]"

                # Cleanup Text
                split_parts = re.split(r"###\s*BẢN DỊCH", trans_text, flags=re.IGNORECASE)
                trans_text = split_parts[1].strip() if len(split_parts) > 1 else trans_text

                if chap_id not in results_map:
                    results_map[chap_id] = {}
                results_map[chap_id][chunk_id] = trans_text

            except Exception as e:
                print(f"   ⚠️ Parsing Error: {e}")
                continue

    os.makedirs(const.OUTPUT_TRANS_DIR, exist_ok=True)
    
    for chap_id, chunks_dict in results_map.items():
        sorted_chunks = sorted(chunks_dict.items())
        full_text = "\n\n".join([text for _, text in sorted_chunks])
        
        out_name = f"trans_chap_{chap_id}_gemini_no.txt"
        out_path = os.path.join(const.OUTPUT_TRANS_DIR, out_name)
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"   ✅ Saved Chapter {chap_id} to {out_name}")

def step_5_delete_job(job_name):
    print(f"🚀 STEP 5: Deleting Batch Job {job_name}...")
    client = genai.Client(api_key=const.GEMINI_API_KEY)
    
    try:
        # Simply pass the job string (e.g., "batches/xyz...")
        client.batches.delete(name=job_name)
        print(f"✅ Successfully deleted job: {job_name}")
    except Exception as e:
        print(f"❌ Failed to delete job. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Added "delete" to choices
    parser.add_argument("mode", choices=["prepare", "submit", "check", "process", "delete"], help="Action to perform")
    parser.add_argument("--job_name", type=str, help="Required for 'check' and 'delete' modes")
    args = parser.parse_args()

    if args.mode == "prepare":
        step_1_prepare_batch_file()
    elif args.mode == "submit":
        step_2_submit_batch()
    elif args.mode == "check":
        if not args.job_name:
            print("⚠️ Please provide --job_name for checking status.")
        else:
            step_3_check_and_download(args.job_name)
    elif args.mode == "process":
        step_4_reassemble_files()
    elif args.mode == "delete":
        if not args.job_name:
            print("⚠️ Please provide --job_name to delete.")
        else:
            step_5_delete_job(args.job_name)
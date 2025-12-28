### File này đang lỗi do có batch job có task thành công task, task fail 


# import os
# import json
# import re
# import time
# import argparse
# from google import genai
# from google.genai import types

# import const
# from modules.prompts import build_evaluation_prompt # Import hàm prompt đánh giá từ file prompts.py

# # --- CẤU HÌNH ĐƯỜNG DẪN CỐ ĐỊNH THEO YÊU CẦU ---
# BASE_PATH = r"C:/Users/trang/OneDrive/Desktop/Dynamic GraphRAG/data/LOTM"
# CHAPTERS_DIR = os.path.join(BASE_PATH, "chapters")
# TRANS_A_DIR = os.path.join(BASE_PATH, "translations", "gemini-2.5-flash_No_RAG")
# TRANS_B_DIR = os.path.join(BASE_PATH, "translations", "gemini-2.5-flash_RAG")

# # Thư mục chứa file batch tạm thời
# BATCH_WORK_DIR = os.path.join(const.DATA_DIR, "batch_evaluation")
# os.makedirs(BATCH_WORK_DIR, exist_ok=True)

# INPUT_JSONL = os.path.join(BATCH_WORK_DIR, "eval_batch_input.jsonl")
# OUTPUT_JSONL = os.path.join(BATCH_WORK_DIR, "eval_batch_output.jsonl")
# RESULT_REPORT = os.path.join(BATCH_WORK_DIR, "evaluation_report.json")

# # Phạm vi chạy
# START_CHAP = 1
# END_CHAP = 10

# def convert_messages_to_gemini_request(messages):
#     """Chuyển đổi format tin nhắn sang định dạng Native Gemini Batch"""
#     contents = []
#     system_instruction = None

#     for msg in messages:
#         role = msg["role"]
#         content_text = msg["content"]
        
#         if role == "system":
#             system_instruction = {"parts": [{"text": content_text}]}
#         elif role == "user":
#             contents.append({"role": "user", "parts": [{"text": content_text}]})
#         elif role == "assistant":
#             contents.append({"role": "model", "parts": [{"text": content_text}]})

#     safety_settings = [
#         {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
#         {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
#         {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
#         {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
#     ]

#     request_body = {
#         "contents": contents,
#         "safetySettings": safety_settings,
#         "generationConfig": {
#             "temperature": 0.05, # Nhiệt độ thấp cho việc đánh giá khách quan
#             "maxOutputTokens": 32000,
#             "responseMimeType": "application/json" # Ép kiểu trả về JSON
#         }
#     }

#     if system_instruction:
#         request_body["system_instruction"] = system_instruction

#     return request_body

# def clean_json_text(text):
#     """Làm sạch markdown json từ response"""
#     cleaned = re.sub(r"```json\s*", "", text)
#     cleaned = re.sub(r"```\s*$", "", cleaned)
#     return cleaned.strip()

# def step_1_prepare_eval_batch():
#     print(f"🚀 STEP 1: Chuẩn bị file Batch Đánh giá (Chap {START_CHAP} - {END_CHAP})...")
    
#     request_buffer = []
#     count = 0

#     for chap_idx in range(START_CHAP, END_CHAP + 1):
#         # 1. Xây dựng đường dẫn file chính xác theo format
#         # Source: chapter_0001.txt
#         src_path = os.path.join(CHAPTERS_DIR, f"chapter_{chap_idx:04d}.txt")
#         # Trans A: trans_chap_1_gemini_no.txt
#         trans_a_path = os.path.join(TRANS_A_DIR, f"trans_chap_{chap_idx}_gemini_no.txt")
#         # Trans B: trans_chap_1_gemini_yes.txt
#         trans_b_path = os.path.join(TRANS_B_DIR, f"trans_chap_{chap_idx}_gemini_yes.txt")

#         # 2. Kiểm tra file tồn tại
#         if not (os.path.exists(src_path) and os.path.exists(trans_a_path) and os.path.exists(trans_b_path)):
#             print(f"   ⚠️ Skipping Chap {chap_idx}: Thiếu một trong các file thành phần.")
#             continue

#         # 3. Đọc nội dung
#         try:
#             with open(src_path, "r", encoding="utf-8") as f: source_text = f.read()
#             with open(trans_a_path, "r", encoding="utf-8") as f: trans_a_text = f.read()
#             with open(trans_b_path, "r", encoding="utf-8") as f: trans_b_text = f.read()
#         except Exception as e:
#             print(f"   ❌ Error reading Chap {chap_idx}: {e}")
#             continue

#         # 4. Tạo Prompt Đánh giá
#         # Gọi hàm từ prompts.py
#         messages = build_evaluation_prompt(source_text, trans_a_text, trans_b_text)
        
#         # 5. Convert sang Gemini Request
#         gemini_request = convert_messages_to_gemini_request(messages)
        
#         # 6. Tạo Batch Item
#         batch_item = {
#             "key": f"eval-chap-{chap_idx}",
#             "request": gemini_request
#         }
        
#         request_buffer.append(json.dumps(batch_item))
#         count += 1
#         print(f"   + Đã thêm Chap {chap_idx} vào hàng đợi.")

#     # Ghi ra file JSONL
#     with open(INPUT_JSONL, "w", encoding="utf-8") as f:
#         f.write("\n".join(request_buffer))
        
#     print(f"✅ Đã tạo {INPUT_JSONL} với {count} yêu cầu đánh giá.")

# def step_2_submit_batch():
#     print("🚀 STEP 2: Upload và Submit Batch Job...")
#     client = genai.Client(api_key=const.GEMINI_API_KEY)
    
#     print(f"   Uploading {INPUT_JSONL}...")
#     batch_file = client.files.upload(
#         file=INPUT_JSONL,
#         config={'mime_type': 'application/json'} 
#     )
    
#     print(f"   Submitting job for model: gemini-3-pro-preview...")
#     batch_job = client.batches.create(
#         model="gemini-3-pro-preview",
#         src=batch_file.name,
#         config=types.CreateBatchJobConfig(
#             display_name=f"eval_chap_{START_CHAP}_{END_CHAP}"
#         )
#     )
    
#     print(f"✅ Job Started! Job Name: {batch_job.name}")
#     print("   LƯU LẠI Job Name để kiểm tra ở bước sau.")
#     return batch_job.name

# def step_3_check_and_download(job_name):
#     print(f"🚀 STEP 3: Kiểm tra trạng thái Job {job_name}...")
#     client = genai.Client(api_key=const.GEMINI_API_KEY)
    
#     try:
#         batch_job = client.batches.get(name=job_name)
#     except Exception as e:
#         print(f"❌ Lỗi lấy thông tin job: {e}")
#         return False
    
#     state_name = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
#     print(f"   Current State: {state_name}")
    
#     if state_name == 'JOB_STATE_SUCCEEDED':
#         print("   Job Finished! Đang tải kết quả...")
        
#         result_file_name = None
#         if hasattr(batch_job, 'output_file') and batch_job.output_file:
#              result_file_name = batch_job.output_file
#         elif hasattr(batch_job, 'dest') and batch_job.dest and hasattr(batch_job.dest, 'file_name'):
#              result_file_name = batch_job.dest.file_name

#         if result_file_name:
#             try:
#                 content = client.files.download(file=result_file_name)
#                 with open(OUTPUT_JSONL, "wb") as f:
#                     f.write(content)
#                 print(f"✅ Kết quả đã lưu tại: {OUTPUT_JSONL}")
#                 return True
#             except Exception as e:
#                 print(f"❌ Lỗi tải file: {e}")
#                 return False
#         else:
#             print("❌ Không tìm thấy tên file đầu ra.")
#             return False

#     elif state_name == 'JOB_STATE_FAILED':
#         print(f"❌ Job Failed. Error: {batch_job.error}")
#         return False
#     else:
#         print(f"   Job đang chạy... Vui lòng thử lại sau.")
#         return False

# def step_4_process_results():
#     print("🚀 STEP 4: Tổng hợp kết quả đánh giá...")
    
#     if not os.path.exists(OUTPUT_JSONL):
#         print("❌ Chưa có file kết quả. Hãy chạy Step 3 trước.")
#         return

#     report_data = []
#     stats = {"A": 0, "B": 0, "tie": 0, "errors": 0}

#     with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip(): continue
#             try:
#                 item = json.loads(line)
#                 custom_id = item.get("key") # format: eval-chap-1
                
#                 # Lấy nội dung response
#                 if "response" in item and "candidates" in item["response"]:
#                     raw_content = item["response"]["candidates"][0]["content"]["parts"][0]["text"]
#                     clean_content = clean_json_text(raw_content)
#                     eval_result = json.loads(clean_content)
                    
#                     winner = eval_result.get("winner", "unknown")
#                     reason = eval_result.get("reason", "No reason provided")
                    
#                     # Update stats
#                     if winner in stats: stats[winner] += 1
                    
#                     report_data.append({
#                         "chapter": custom_id,
#                         "winner": winner,
#                         "reason": reason
#                     })
#                 else:
#                     stats["errors"] += 1
#                     print(f"⚠️ Error in response for {custom_id}")

#             except Exception as e:
#                 print(f"⚠️ Parsing Error: {e}")
#                 continue
    
#     # Sắp xếp report theo chapter
#     report_data.sort(key=lambda x: int(x["chapter"].split("-")[-1]))

#     # Ghi file báo cáo tổng hợp
#     full_report = {
#         "summary_stats": stats,
#         "details": report_data
#     }
    
#     with open(RESULT_REPORT, "w", encoding="utf-8") as f:
#         json.dump(full_report, f, indent=4, ensure_ascii=False)

#     print("\n------------------------------------------------")
#     print("📊 TỔNG KẾT ĐÁNH GIÁ:")
#     print(f"   - Tổng số chương: {len(report_data)}")
#     print(f"   - Thắng (A - No RAG): {stats['A']}")
#     print(f"   - Thắng (B - RAG):    {stats['B']}")
#     print(f"   - Hòa (Tie):          {stats['tie']}")
#     print(f"✅ Chi tiết lưu tại: {RESULT_REPORT}")
#     print("------------------------------------------------")

# def step_5_delete_job(job_name):
#     print(f"🚀 STEP 5: Xóa Job {job_name}...")
#     client = genai.Client(api_key=const.GEMINI_API_KEY)
#     try:
#         client.batches.delete(name=job_name)
#         print(f"✅ Đã xóa job thành công.")
#     except Exception as e:
#         print(f"❌ Xóa thất bại: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("mode", choices=["prepare", "submit", "check", "process", "delete"], help="Chọn chế độ chạy")
#     parser.add_argument("--job_name", type=str, help="Tên Job (cần cho check/delete)")
    
#     args = parser.parse_args()

#     if args.mode == "prepare":
#         step_1_prepare_eval_batch()
#     elif args.mode == "submit":
#         step_2_submit_batch()
#     elif args.mode == "check":
#         if args.job_name: step_3_check_and_download(args.job_name)
#         else: print("⚠️ Cần tham số --job_name")
#     elif args.mode == "process":
#         step_4_process_results()
#     elif args.mode == "delete":
#         if args.job_name: step_5_delete_job(args.job_name)
#         else: print("⚠️ Cần tham số --job_name")
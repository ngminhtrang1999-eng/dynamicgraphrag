import os
import pandas as pd
import torch
import re
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import multiprocessing

# Configurations

OUTPUT_DIR = "evaluation/alignment_results"
MODEL_NAME = 'sentence-transformers/LaBSE' 

# Global variables for worker processes
model = None
device = None

def init_worker():
    """Initializer for worker processes to load model once."""
    global model, device
    print(f"--- Process {os.getpid()} initializing ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{os.getpid()}] Device: {device.upper()}")
    print(f"[{os.getpid()}] Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    model.to(device)
    print(f"[{os.getpid()}] Model loaded.")

# --- 2. HÀM TÁCH CÂU (ROBUST SPLIT) ---
def smart_sentence_split(text):
    """
    Tách câu dựa trên quy tắc Regex mạnh, hoạt động tốt cho cả Anh và Việt
    mà không cần phụ thuộc vào Spacy Model (tránh lỗi không load được model).
    """
    if isinstance(text, list):
        return text # Đã là list thì trả về luôn

    # 1. Làm sạch văn bản thô
    text = str(text)
    text = text.replace('…', '...').replace('\r\n', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. Tách câu dựa trên dấu kết thúc (. ? !) nhưng có nhìn trước (Lookahead)
    # Logic: Tách khi gặp dấu kết thúc câu, theo sau là khoảng trắng và ký tự in hoa hoặc số.
    # Pattern này bảo vệ các từ viết tắt phổ biến như Mr., Dr. (tương đối)
    pattern = r'(?<=[.?!])\s+(?=[A-Z"“0-9])'
    
    sentences = re.split(pattern, text)
    
    # Lọc bỏ các câu quá ngắn hoặc rỗng
    return [s.strip() for s in sentences if len(s.strip()) > 1]

# --- 3. THUẬT TOÁN GLOBAL DYNAMIC ALIGNMENT (CORE) ---
def global_dynamic_alignment(src_list, tgt_list, max_merge_tgt=5, max_merge_src=5): # <--- TĂNG LÊN 5
    """
    Phiên bản V2: Tối ưu cho hội thoại ngắn và giảm thiểu lệch dòng.
    """
    # Verify model is loaded (should be done by init_worker)
    if model is None:
        raise RuntimeError("Model not initialized in worker process")

    n = len(src_list)
    m = len(tgt_list)
    
    # Encode vectors
    src_emb = model.encode(src_list, convert_to_tensor=True, device=device, show_progress_bar=False)
    tgt_emb = model.encode(tgt_list, convert_to_tensor=True, device=device, show_progress_bar=False)
    
    src_emb = F.normalize(src_emb, p=2, dim=1)
    tgt_emb = F.normalize(tgt_emb, p=2, dim=1)

    dp = np.full((n + 1, m + 1), -np.inf)
    dp[0][0] = 0
    trace = {} 

    # Trọng số phạt (Penalty):
    # Mỗi lần gộp thêm 1 câu, trừ nhẹ điểm số để ưu tiên khớp 1-1 nếu điểm tương đồng ngang nhau.
    # Penalty cho source nên cao hơn chút vì source thường chuẩn hơn.
    PENALTY_MERGE_TGT = 0.05 
    PENALTY_MERGE_SRC = 0.1  

    # print(f"[{os.getpid()}] Calculating alignment...")
    
    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == -np.inf: continue
            
            # --- ACTION 1: 1 Source - K Target (Source dài, Trans ngắt dòng) ---
            if i < n:
                current_src_vec = src_emb[i]
                accumulated_tgt_vec = torch.zeros_like(tgt_emb[0])
                
                for k in range(1, max_merge_tgt + 1):
                    if j + k > m: break
                    
                    accumulated_tgt_vec += tgt_emb[j + k - 1]
                    norm_tgt = F.normalize(accumulated_tgt_vec, p=2, dim=0)
                    score = torch.dot(current_src_vec, norm_tgt).item()
                    
                    # Áp dụng phạt: Trừ điểm nếu gộp nhiều
                    penalty = PENALTY_MERGE_TGT * (k - 1)
                    new_score = dp[i][j] + score - penalty
                    
                    if new_score > dp[i+1][j+k]:
                        dp[i+1][j+k] = new_score
                        trace[(i+1, j+k)] = (i, j, f"1-{k}")

            # --- ACTION 2: K Source - 1 Target (Source ngắn, Trans gộp) ---
            # ĐÂY LÀ PHẦN SỬA LỖI CỦA BẠN
            if j < m:
                current_tgt_vec = tgt_emb[j]
                accumulated_src_vec = torch.zeros_like(src_emb[0])
                
                for k in range(1, max_merge_src + 1):
                    if i + k > n: break
                    
                    # Skip k=1 ở đây nếu muốn (vì trùng Action 1) nhưng để logic riêng cũng không sao
                    # Code cũ bỏ qua k=1, nhưng code mới nên giữ để tính penalty đồng nhất
                    if k == 1 and i < n: 
                        # Đã tính ở Action 1 (dạng 1-1), nhưng ta cần check xem
                        # liệu đi hướng này (coi là gộp src) có tốt hơn không (thường là bằng nhau)
                        pass 

                    accumulated_src_vec += src_emb[i + k - 1]
                    norm_src = F.normalize(accumulated_src_vec, p=2, dim=0)
                    score = torch.dot(norm_src, current_tgt_vec).item()
                    
                    # Phạt nặng hơn chút nếu gộp nhiều Source
                    penalty = PENALTY_MERGE_SRC * (k - 1)
                    new_score = dp[i][j] + score - penalty
                    
                    if new_score > dp[i+k][j+1]:
                        dp[i+k][j+1] = new_score
                        trace[(i+k, j+1)] = (i, j, f"{k}-1")

    # --- Backtracking (Giữ nguyên) ---
    aligned_pairs = []
    curr_i, curr_j = n, m
    
    # Fallback tìm đích đến tốt nhất nếu không chạm (n, m)
    if (curr_i, curr_j) not in trace:
        # print(f"[{os.getpid()}] Warning: Path lost, using best endpoint...")
        max_score = -np.inf
        best_pos = (n, m)
        # Quét hàng cuối và cột cuối
        for x in range(n + 1):
            if dp[x][m] > max_score: max_score, best_pos = dp[x][m], (x, m)
        for y in range(m + 1):
            if dp[n][y] > max_score: max_score, best_pos = dp[n][y], (n, y)
        curr_i, curr_j = best_pos

    while curr_i > 0 and curr_j > 0:
        if (curr_i, curr_j) not in trace:
             # Xử lý kẹt: nhảy về trước 1 bước bất kỳ
            if curr_i > 0: curr_i -= 1
            elif curr_j > 0: curr_j -= 1
            continue

        prev_i, prev_j, match_type = trace[(curr_i, curr_j)]
        
        src_text = " ".join(src_list[prev_i:curr_i])
        tgt_text = " ".join(tgt_list[prev_j:curr_j])
        
        aligned_pairs.append({
            "Ref": src_text,
            "Trans": tgt_text,
            "Type": match_type
        })
        curr_i, curr_j = prev_i, prev_j

    aligned_pairs.reverse()
    return aligned_pairs

def create_master_csv(src_sents, dict_alignments):
    """
    Tạo bảng Master CSV lấy Source làm chuẩn.
    """
    
    # 1. Khởi tạo khung dữ liệu chuẩn với Source
    master_data = []
    for idx, text in enumerate(src_sents):
        row = {"Source_ID": idx, "Source_Text": text}
        # Khởi tạo cột cho các bản dịch là None
        for key in dict_alignments.keys():
            row[key] = None 
        master_data.append(row)

    # 2. Duyệt qua từng bản dịch để điền vào khung
    for model_name, aligned_pairs in dict_alignments.items():
        
        # Con trỏ hiện tại trong danh sách aligned_pairs
        pair_idx = 0
        
        # Con trỏ hiện tại trong danh sách Source gốc
        src_idx = 0
        
        while src_idx < len(src_sents) and pair_idx < len(aligned_pairs):
            pair = aligned_pairs[pair_idx]
            ref_text_in_pair = pair['Ref'] # Text source trong cặp align (có thể là 1 hoặc nhiều câu gộp)
            trans_text = pair['Trans']
            match_type = pair['Type'] # vd: "1-1", "2-1"
            
            # --- LOGIC QUAN TRỌNG: MAPPING ---
            num_src_merged = int(match_type.split('-')[0])
            
            # Điền dữ liệu vào dòng đầu tiên của block (Anchor)
            if src_idx < len(master_data):
                master_data[src_idx][model_name] = trans_text
            
            # Đánh dấu các dòng bị gộp phía sau
            for k in range(1, num_src_merged):
                if src_idx + k < len(master_data):
                    master_data[src_idx + k][model_name] = "<MERGED_ABOVE>"
            
            # Cập nhật con trỏ
            src_idx += num_src_merged
            pair_idx += 1
            
    return pd.DataFrame(master_data)

def process_chapter(i):
    print(f"[{os.getpid()}] Processing Chapter {i}...")
    
    # Define File Paths
    path_src = f"data/LOTM/chapters/chapter_{i:04d}.txt" 
    path_ref = f"data/LOTM/translations/reference/ref_{i}.txt"
    path_rag = f"data/LOTM/translations/gemini-2.5-pro_RAG/trans_chap_{i}_gemini2.5-pro_yes.txt"
    path_no_rag = f"data/LOTM/translations/gemini-2.5-pro_No_RAG/trans_chap_{i}_gemini2.5-pro_no.txt"
    path_gg = f"data/LOTM/translations/google_translate/translated_chapter_{i:04d}.txt"
    
    # Verify files exist
    missing_files = []
    if not os.path.exists(path_src): missing_files.append(path_src)
    if not os.path.exists(path_ref): missing_files.append(path_ref)
    if not os.path.exists(path_rag): missing_files.append(path_rag)
    if not os.path.exists(path_no_rag): missing_files.append(path_no_rag)
    if not os.path.exists(path_gg): missing_files.append(path_gg)
    
    if missing_files:
        print(f"[{os.getpid()}] SKIP Chapter {i}: Missing files: {missing_files}")
        return

    try:
        # Read files
        with open(path_src, 'r', encoding='utf-8-sig') as f: src_text = f.read()
        with open(path_ref, 'r', encoding='utf-8-sig') as f: ref_text = f.read()
        with open(path_rag, 'r', encoding='utf-8-sig') as f: rag_text = f.read()
        with open(path_no_rag, 'r', encoding='utf-8-sig') as f: no_rag_text = f.read()
        with open(path_gg, 'r', encoding='utf-8-sig') as f: gg_text = f.read()
        
        # Segment sentences
        src_sents = smart_sentence_split(src_text)
        ref_sents = smart_sentence_split(ref_text)
        rag_sents = smart_sentence_split(rag_text)
        no_rag_sents = smart_sentence_split(no_rag_text)
        gg_sents = smart_sentence_split(gg_text)
        
        # --- Alignment ---
        # print(f"[{os.getpid()}] Aligning Chapter {i}...")
        res_ref = global_dynamic_alignment(src_sents, ref_sents, max_merge_tgt=5)
        res_rag = global_dynamic_alignment(src_sents, rag_sents, max_merge_tgt=5)
        res_no_rag = global_dynamic_alignment(src_sents, no_rag_sents, max_merge_tgt=5)
        res_gg = global_dynamic_alignment(src_sents, gg_sents, max_merge_tgt=5)
        
        # --- Create Master CSV ---
        df_master = create_master_csv(
            src_sents, 
            {
                "ref": res_ref, 
                "RAG": res_rag, 
                "No_RAG": res_no_rag, 
                "GG": res_gg
            }
        )
        
        out_path = os.path.join(OUTPUT_DIR, f"aligned_master_chap_{i}.csv")
        df_master.to_csv(out_path, index=False, encoding='utf-8-sig')
        
        print(f"[{os.getpid()}] DONE Chapter {i}. Saved to {out_path}")

    except Exception as e:
        print(f"[{os.getpid()}] ERROR processing Chapter {i}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print(f"Output Directory: {OUTPUT_DIR}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Use 3 processes as requested
    num_processes = 2
    chapters = list(range(1, 11))

    print(f"Starting pool with {num_processes} processes for {len(chapters)} chapters...")
    
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker) as pool:
        pool.map(process_chapter, chapters)

    print("\nAll chapters processed.")

if __name__ == "__main__":
    # Ensure this is safe for Windows/Spawn
    multiprocessing.set_start_method('spawn', force=True)
    main()

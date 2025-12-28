import os
import time
import spacy
# --- THAY THẾ IMPORT: Thay googletrans bằng deep_translator ---
from deep_translator import GoogleTranslator
import const
import re

# --- KHỞI TẠO SPACY ---
print("Đang tải model Spacy...")

# Tải model spaCy
nlp = spacy.load("en_core_web_trf")

# Tạo folder output (xử lý lỗi No such file or directory)
os.makedirs("data/LOTM/translations/google_translate", exist_ok=True)

# --- KHỞI TẠO DỊCH VỚI DEEP_TRANSLATOR (GoogleTranslator) ---
# deep_translator không cần tạo đối tượng Translator()
# mà gọi trực tiếp phương thức translate() trên lớp GoogleTranslator
# Tuy nhiên, để tiện cho việc sử dụng, ta có thể khởi tạo trước
# GoogleTranslator(source='en', target='vi')

all_files = os.listdir(const.INPUT_DIR)

# --- HÀM TÁCH TEXT MỚI (CỦA BẠN) ---
# (Hàm này giữ nguyên vì nó là logic tách câu độc lập với thư viện dịch)
def split_text_with_overlap(text, max_words=300, overlap_ratio=0.0):
    """
    Chia text thành các chunk dựa trên số từ, đảm bảo trọn vẹn câu.
    LƯU Ý: Nếu dùng overlap_ratio > 0 cho Google Translate, bản dịch sẽ bị LẶP LẠI đoạn gối đầu.
    Tôi để mặc định overlap_ratio=0 để tránh lặp câu khi dịch.
    """
    # Tăng giới hạn ký tự cho spacy
    nlp.max_length = len(text) + 100000
    doc = nlp(text)
    
    sentences = list(doc.sents)
    n_sents = len(sentences)
    
    chunks = []
    start_idx = 0
    
    while start_idx < n_sents:
        current_word_count = 0
        end_idx = start_idx
        
        # 1. Forward: Gom câu
        while end_idx < n_sents:
            sent_len = len(sentences[end_idx]) # Số token
            # Nếu chunk hiện tại đã có dữ liệu và thêm câu này bị vượt quá max_words -> dừng
            if current_word_count + sent_len > max_words and current_word_count > 0:
                break
            
            current_word_count += sent_len
            end_idx += 1
        
        # Tạo text cho chunk
        chunk_text = "".join([sent.text_with_ws for sent in sentences[start_idx:end_idx]])
        chunks.append(chunk_text)
        
        if end_idx >= n_sents:
            break
            
        # 2. Backward: Tính overlap (Giữ nguyên logic nhưng chú thích là không nên dùng)
        if overlap_ratio > 0:
            overlap_target = int(current_word_count * overlap_ratio)
            overlap_accumulated = 0
            new_start_idx = end_idx
            
            while new_start_idx > start_idx:
                prev_sent_len = len(sentences[new_start_idx - 1])
                if overlap_accumulated + prev_sent_len > overlap_target:
                    break
                overlap_accumulated += prev_sent_len
                new_start_idx -= 1
                
            if new_start_idx <= start_idx:
                new_start_idx = start_idx + 1
            start_idx = new_start_idx
        else:
            # Nếu không dùng overlap thì đi tiếp từ vị trí kết thúc cũ
            start_idx = end_idx

    return chunks

# --- MAIN LOOP ---
for chap_idx in range(const.START_CHAPTER, const.END_CHAPTER + 1):
    # Tìm file với logic chính xác hơn
    file_name = next((f for f in all_files if f"{chap_idx}." in f or f"_{chap_idx}." in f), None)
    
    # Fallback: nếu không tìm thấy theo cách chuẩn, tìm theo cách cũ
    if not file_name:
         file_name = next((f for f in all_files if str(chap_idx) in f), None)

    if not file_name:
        print(f"Bỏ qua chương {chap_idx} (Không tìm thấy file)")
        continue
        
    input_path = os.path.join(const.INPUT_DIR, file_name)
    # --- THAY ĐỔI OUTPUT PATH: Đổi tên folder để phân biệt ---
    output_path = os.path.join("data/LOTM/translations/google_translate", f"translated_{file_name}")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = re.sub(r'\s+', ' ', text)

        if not text.strip(): continue

        print(f"Đang xử lý: {file_name}...")

        # --- GỌI HÀM TÁCH TEXT MỚI ---
        # Giữ nguyên các tham số tách (max_words=1200, overlap_ratio=0)
        chunks = split_text_with_overlap(text, max_words=300, overlap_ratio=0)
        
        translated_parts = []
        for i, chunk in enumerate(chunks):
            try:
                # --- THAY ĐỔI HÀM DỊCH: Sử dụng deep_translator.GoogleTranslator.translate_batch/translate ---
                # Vì ta đang dịch từng chunk, dùng translate() là phù hợp.
                res = GoogleTranslator(source='en', target='vi').translate(chunk)
                translated_parts.append(res) # res là chuỗi đã dịch

                # Sleep nhẹ giữa các chunk để tránh spam
                time.sleep(0.5) 
            except Exception as e:
                print(f"  Lỗi chunk {i} (deep_translator): {e}")
                translated_parts.append(chunk) # Giữ nguyên nếu lỗi

        final_text = " ".join(translated_parts)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)
            
        print(f"-> Xong: {file_name}")
        time.sleep(1) # Nghỉ giữa các file

    except Exception as e:
        print(f"Lỗi file {file_name}: {e}")
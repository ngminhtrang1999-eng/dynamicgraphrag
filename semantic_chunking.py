import semchunk
import tiktoken
import os
import re
from utils import split_text_with_overlap
# --- BƯỚC 2: Đọc file text ---
def read_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

text = read_text_file("data/LOTM/chapters/chapter_0008.txt")
text = re.sub(r'\s+', ' ', text)

# --- BƯỚC 3: Cấu hình chunker ---
# 1. Chọn kích thước chunk (ví dụ: 30 token). 
# Trong thực tế, bạn thường dùng 512 hoặc 1024 token.
chunk_size = 300 

# 2. Định nghĩa cách đếm token. 
# Cách chuyên nghiệp: Dùng tiktoken (chuẩn của OpenAI)
try:
    encoder = tiktoken.get_encoding("cl100k_base") # Encoding dùng cho GPT-4
    token_counter = lambda x: len(encoder.encode(x))
    print("ℹ️  Đang sử dụng bộ đếm token của OpenAI (tiktoken).")
except ImportError:
    # Cách đơn giản (nếu không cài tiktoken): Đếm số từ bằng cách tách khoảng trắng
    token_counter = lambda x: len(x.split())
    print("ℹ️  Đang sử dụng bộ đếm từ đơn giản (split space).")

# --- BƯỚC 4: Thực hiện Chunking ---
# semchunk tự động tìm điểm cắt hợp lý nhất (ví dụ: dấu chấm câu) để không làm gãy câu.
chunks = semchunk.chunk(text, chunk_size=chunk_size, token_counter=token_counter)

# --- BƯỚC 5: In kết quả ---
print(f"\n--- KẾT QUẢ CHUNKING (Size: {chunk_size} tokens) ---")
print(f"Tổng số chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    token_count = token_counter(chunk)
    print(f"🔹 Chunk {i+1} ({token_count} tokens):")
    print(f"\"{chunk}\"")
    print("-" * 40)



import os
import re

# ---- SETTINGS ----
input_folder = r"data\LOTM"             # Thư mục chứa các file Vol 1, Vol 2...
output_dir = r"data\LOTM\chapters"      # Thư mục lưu kết quả
# -------------------

def get_volume_number(filename):
    """
    Hàm lấy số Volume từ tên file để sắp xếp cho đúng.
    VD: "Clown - LotM Vol. 1_djvu.txt" -> Trả về 1
    VD: "Side Stories - LotM Vol. 9_djvu.txt" -> Trả về 9
    """
    match = re.search(r"Vol\.\s*(\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 999 # Nếu không tìm thấy số, đẩy xuống cuối

def clean_novel_content(text):
    if not text: return ""
    text = text.replace('\u200b', '')
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n\n'.join(lines)

def extract_clean_title(header_line):
    """Lấy tiêu đề, bỏ qua phần số lỗi"""
    header = header_line.strip()
    if ':' in header:
        return header.split(':', 1)[1].strip()
    parts = header.split(' ', 2)
    if len(parts) > 2:
        return parts[2].strip()
    return ""

# Tạo thư mục output
os.makedirs(output_dir, exist_ok=True)

# 1. LẤY DANH SÁCH FILE VÀ SẮP XẾP
files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

# Sắp xếp file dựa trên số Volume tìm được trong tên file
# (Quan trọng: Để đảm bảo Vol 1 chạy trước Vol 2, v.v...)
files.sort(key=get_volume_number)

print(f"Tìm thấy {len(files)} file volumes. Thứ tự xử lý:")
for f in files:
    print(f" -> {f}")
print("-" * 30)

# 2. XỬ LÝ TUẦN TỰ (GLOBAL COUNTER)
global_chapter_idx = 1 # Bắt đầu đếm từ 1 và tăng dần qua CÁC FILE

pattern = r"(Chapter.*?\n)(.*?)(?=\nChapter|$)"

for filename in files:
    file_path = os.path.join(input_folder, filename)
    print(f"Đang xử lý file: {filename}...")
    
    # Đọc nội dung file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()

    # Tìm tất cả chapter trong file này
    matches = re.findall(pattern, text, flags=re.S)
    
    if not matches:
        print(f"⚠️ Cảnh báo: Không tìm thấy chapter nào trong {filename}")
        continue

    # Duyệt qua từng chapter tìm được trong file hiện tại
    for raw_header, content in matches:
        # Lấy tiêu đề gốc
        original_title = extract_clean_title(raw_header)
        if not original_title:
            original_title = "No Title"

        # Clean nội dung
        clean_content = clean_novel_content(content)

        # Tạo tên file output (Dùng 4 chữ số: 0001, 0002... vì LOTM có hơn 1000 chương)
        out_filename = f"chapter_{global_chapter_idx:04d}.txt"
        out_filepath = os.path.join(output_dir, out_filename)

        # Ghi file
        with open(out_filepath, "w", encoding="utf-8") as out:
            # Format lại header: Chapter [Số đếm toàn cục]: [Tiêu đề gốc]
            out.write(f"Chapter {global_chapter_idx}: {original_title}\n\n")
            out.write(clean_content)
            
        # Tăng biến đếm toàn cục lên 1
        global_chapter_idx += 1

print("-" * 30)
print(f"✅ HOÀN TẤT TOÀN BỘ!")
print(f"Tổng số chương đã tạo: {global_chapter_idx - 1}")
print(f"Thư mục lưu: {output_dir}")
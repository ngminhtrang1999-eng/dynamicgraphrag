import re

def read_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

text = read_text_file("data/LOTM/chapters/chapter_0003.txt")
text = re.sub(r'\s+', ' ', text)

def add_placeholder_regex(text):
    # Danh sách các pronouns bạn muốn nhắm tới
    # (?i) là cờ để không phân biệt hoa thường (He = he)
    # \b là ranh giới từ (đảm bảo "he" đứng riêng lẻ, không nằm trong "the")
    pattern = r"(?i)\b(he|she|him|her|his)\b"
    
    # Thay thế từ tìm được bằng: Chính nó (\g<0>) + dấu cách + []
    modified_text = re.sub(pattern, r"\g<0> []", text)
    
    return modified_text

result = add_placeholder_regex(text)

print(result)
# Output:
# Klein is smart. He [] has a good calculation and with his [] combat skills made him [] very good.
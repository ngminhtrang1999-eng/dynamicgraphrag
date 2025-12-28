import spacy
import networkx as nx
import os
import re
from pyvis.network import Network
from collections import Counter

# ==========================================
# CẤU HÌNH & KHỞI TẠO
# ==========================================

INPUT_DIR = "data\LOTM\chapters" 
OUTPUT_FOLDER = "ket_qua_pyvis_dynamic"

BATCH_SIZE = 25       # Số chapter gộp mỗi lần
OVERLAP_RATIO = 0.15   # Tỷ lệ gối đầu (20%)
WINDOW_SIZE = 20      # Khoảng cách sliding window
MIN_OCCURRENCES = 1   # <--- MỚI: Chỉ hiện nhân vật xuất hiện >= 5 lần

# --- HÀM LOAD MODEL THÔNG MINH (MỚI) ---
def load_model_smart():
    """Kiểm tra GPU và load model phù hợp"""
    return spacy.load("en_core_web_trf")



# Khởi tạo model
nlp = load_model_smart()

# ==========================================
# CÁC HÀM XỬ LÝ NLP & GRAPH
# ==========================================

def auto_unify_characters(raw_names_list):
    """Gộp tên nhân vật (Ví dụ: Klein -> Klein Moretti)"""
    name_counts = Counter(raw_names_list)
    unique_names = sorted(name_counts.keys(), key=lambda x: len(x), reverse=True)
    mapping = {}
    
    for name in unique_names:
        mapping[name] = name
        for potential_parent in unique_names:
            if len(potential_parent) <= len(name):
                break
            parts_short = set(name.split())
            parts_long = set(potential_parent.split())
            if parts_short.issubset(parts_long):
                mapping[name] = potential_parent
                break
    return mapping

def build_character_network_auto(text):
    """Xây dựng đồ thị từ văn bản với bộ lọc tần suất"""
    nlp.max_length = len(text) + 100000 
    doc = nlp(text)
    
    # 1. Lấy tất cả thực thể thô
    raw_entities = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            clean_name = ent.text.strip().replace("'s", "").replace("’s", "").replace("\n", " ")
            if len(clean_name) > 1:
                raw_entities.append({"name": clean_name, "start": ent.start})

    if not raw_entities:
        return nx.Graph()

    # 2. Gộp tên (Canonicalization)
    all_name_strings = [e["name"] for e in raw_entities]
    unified_map = auto_unify_characters(all_name_strings)
    
    # 3. Đếm tần suất tên ĐÃ CHUẨN HÓA (Logic Mới)
    # Ta phải đếm sau khi gộp. Ví dụ: "Klein" (2 lần) + "Klein Moretti" (3 lần) = 5 lần -> Đủ điều kiện.
    canonical_counts = Counter()
    for raw_name in all_name_strings:
        canonical_name = unified_map[raw_name]
        canonical_counts[canonical_name] += 1
    
    # Tạo set các tên hợp lệ (xuất hiện >= MIN_OCCURRENCES)
    valid_names = {name for name, count in canonical_counts.items() if count >= MIN_OCCURRENCES}
    
    print(f"   -> Lọc thực thể: Giữ lại {len(valid_names)} nhân vật (xuất hiện >= {MIN_OCCURRENCES} lần).")

    # 4. Xây dựng đồ thị chỉ với các tên hợp lệ
    G = nx.Graph()
    final_entities = []
    
    for item in raw_entities:
        canonical_name = unified_map[item["name"]]
        
        # --- BƯỚC LỌC QUAN TRỌNG ---
        if canonical_name in valid_names:
            final_entities.append({"name": canonical_name, "start": item["start"]})
            G.add_node(canonical_name)

    # 5. Sliding Window cho Edge
    for i in range(len(final_entities)):
        for j in range(i + 1, len(final_entities)):
            person_a = final_entities[i]
            person_b = final_entities[j]
            distance = person_b["start"] - person_a["start"]
            
            if distance <= WINDOW_SIZE:
                u, v = person_a["name"], person_b["name"]
                if u != v:
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1)
            else:
                break
    return G

# ==========================================
# CÁC HÀM HỖ TRỢ FILE (GIỮ NGUYÊN)
# ==========================================

def read_and_sort_chapters(directory):
    if not os.path.exists(directory): return []
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    def extract_number(filename):
        s = re.findall(r'\d+', filename)
        return int(s[0]) if s else 0
    files.sort(key=extract_number)
    return [os.path.join(directory, f) for f in files]

def get_safe_overlap_text(text, ratio=0.2):
    total_len = len(text)
    if total_len == 0: return ""
    target_start_index = int(total_len * (1 - ratio))
    safe_index = -1
    scan_range = 2000 
    start_scan = max(0, target_start_index - scan_range)
    segment_to_scan = text[start_scan:target_start_index]
    matches = list(re.finditer(r'[.!?\n]', segment_to_scan))
    if matches:
        last_match = matches[-1]
        safe_index = start_scan + last_match.end()
    else:
        safe_index = text.find(' ', target_start_index)
        if safe_index == -1: safe_index = target_start_index
    return text[safe_index:].strip()

# ==========================================
# LOGIC CHÍNH
# ==========================================

print("1. Đang đọc dữ liệu chapters...")
chapter_files = read_and_sort_chapters(INPUT_DIR)

# Demo mode nếu không có file
if not chapter_files:
    print("   -> (Demo Mode) Tạo dữ liệu giả lập...")
    os.makedirs(INPUT_DIR, exist_ok=True)
    for i in range(1, 101):
        with open(os.path.join(INPUT_DIR, f"chap_{i}.txt"), "w", encoding="utf-8") as f:
            content = f"Chapter {i}. Sherlock Holmes talked to Watson. " * 20
            # Thêm nhân vật rác xuất hiện ít để test bộ lọc
            content += " A random passerby appeared once. " 
            if i % 25 == 0: content += " Moriarty appeared. "
            f.write(content)
    chapter_files = read_and_sort_chapters(INPUT_DIR)

print(f"   -> Tìm thấy {len(chapter_files)} chapters.")

dynamic_graphs = []
previous_overlap_text = ""

total_chapters = len(chapter_files)
for i in range(0, total_chapters, BATCH_SIZE):
    batch_files = chapter_files[i : i + BATCH_SIZE]
    batch_id = (i // BATCH_SIZE) + 1
    
    print(f"\n--- Xử lý Batch {batch_id} (Chapters {i+1} đến {min(i+BATCH_SIZE, total_chapters)}) ---")
    
    current_batch_content = ""
    for file_path in batch_files:
        with open(file_path, "r", encoding="utf-8") as f:
            current_batch_content += f.read() + "\n"
    
    full_text_to_process = ""
    if previous_overlap_text:
        full_text_to_process = previous_overlap_text + "\n" + current_batch_content
    else:
        full_text_to_process = current_batch_content

    print(f"   + Đang chạy spaCy (Length: {len(full_text_to_process)} chars)...")
    g_batch = build_character_network_auto(full_text_to_process)
    
    # Chỉ thêm vào danh sách nếu đồ thị có node (sau khi đã lọc)
    if len(g_batch.nodes) > 0:
        dynamic_graphs.append(g_batch)
    
    previous_overlap_text = get_safe_overlap_text(full_text_to_process, ratio=OVERLAP_RATIO)

print(f"\n-> Đã tạo thành công {len(dynamic_graphs)} đồ thị.")

# ==========================================
# LƯU FILE HTML (ĐÃ FIX LỖI CHARMAP)
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"\n2. Bắt đầu lưu file HTML vào thư mục '{OUTPUT_FOLDER}'...")

for i, G_original in enumerate(dynamic_graphs):
    G_vis = G_original.copy()
    
    degrees = dict(nx.degree(G_vis))
    # Tránh chia cho 0 nếu đồ thị rỗng (dù hiếm khi xảy ra do check ở trên)
    max_degree = max(degrees.values()) if degrees else 1
    
    for node_id in G_vis.nodes():
        size = 10 + (degrees[node_id] / max_degree) * 20  
        G_vis.nodes[node_id]['size'] = size
        G_vis.nodes[node_id]['color'] = '#97C2FC'
        G_vis.nodes[node_id]['title'] = f"Nhân vật: {node_id}\nSố kết nối: {degrees[node_id]}"

    for u, v, data in G_vis.edges(data=True):
        weight = data.get('weight', 1)
        data['value'] = weight 
        data['title'] = f"Tương tác: {weight}"

    net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='in_line', notebook=False)
    net.from_nx(G_vis)
    net.show_buttons(filter_=['physics'])

    file_name = f"graph_batch_{i+1:03d}.html"
    full_path = os.path.join(OUTPUT_FOLDER, file_name)
    
    try:
        # FIX LỖI CHARMAP: Generate HTML trước rồi ghi bằng utf-8
        html_content = net.generate_html()
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"   -> Đã lưu: {file_name} (Nodes: {len(G_vis.nodes)})")
    except Exception as e:
        print(f"Lỗi lưu file {file_name}: {e}")

print(f"\nHOÀN TẤT! Mở folder '{OUTPUT_FOLDER}' để xem.")
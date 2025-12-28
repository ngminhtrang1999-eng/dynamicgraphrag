import os
import json
import re
from core.llm_backend import GeminiAPILLM
from modules.prompts import build_evaluation_prompt
# Giả sử file const.py đã có và chứa API KEY

def read_file(file_path):
    """Hàm đọc file text an toàn với encoding UTF-8"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def clean_json_response(response_text):
    """
    Làm sạch phản hồi từ LLM để lấy chuỗi JSON hợp lệ.
    Thường LLM sẽ trả về dạng ```json { ... } ```
    """
    # Xóa markdown code block nếu có
    cleaned = re.sub(r"```json\s*", "", response_text)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    return cleaned.strip()

def main():
    # --- CẤU HÌNH ĐƯỜNG DẪN FILE ---
    # Bạn có thể thay đổi tên file tương ứng ở đây
    source_path = "data/LOTM/chapters/chapter_0009.txt"       # Văn bản gốc (Tiếng Anh)
    trans_a_path = "data/LOTM/translations/llama-3.1-8B_No_RAG/trans_chap_9_llama3.1-8b_no.txt"     # Bản dịch A
    trans_b_path = "data/LOTM/translations/llama-3.1-8B_RAG/trans_chap_9_llama3.1-8b_yes.txt"     # Bản dịch B

    print("--- BẮT ĐẦU QUÁ TRÌNH ĐÁNH GIÁ ---")

    try:
        # 1. Load nội dung file m
        print(f"📂 Đang đọc dữ liệu từ các file...")
        source_text = read_file(source_path)
        trans_a = read_file(trans_a_path)
        trans_b = read_file(trans_b_path)
        
        print(f"✅ Đã đọc xong. Độ dài Source: {len(source_text)} ký tự.")

        # 2. Khởi tạo LLM (Gemini)
        # Lưu ý: Đảm bảo const.py đã có GEMINI_API_KEY
        llm = GeminiAPILLM(model_name="gemini-3-pro-preview")

        # 3. Tạo Prompt
        # Sử dụng hàm build_evaluation_prompt từ prompts.py
        messages = build_evaluation_prompt(source_text, trans_a, trans_b)

        # 4. Gửi yêu cầu đánh giá
        print("🤖 Đang gửi yêu cầu tới Gemini... Vui lòng chờ.")
        # Nhiệt độ thấp (0.1 - 0.2) để đánh giá khách quan, ít sáng tạo
        response_text = llm.generate(messages, temperature=0.05)
        print(response_text)

        # print("--- KIỂM TRA PHẢN HỒI ---")
        # candidate = response_text.candidates[0]

        # # 1. Kiểm tra Finish Reason
        # print(f"Lý do dừng (Finish Reason): {candidate.finish_reason}")

        # # 2. Kiểm tra xem có Function Call không? (QUAN TRỌNG NHẤT)
        # if candidate.content.parts:
        #     for part in candidate.content.parts:
        #         if part.function_call:
        #             print(f"⚠️ ĐÂY LÀ FUNCTION CALL (không phải text): {part.function_call}")
        #         if part.text:
        #             print(f"Văn bản: {part.text}")
        # else:
        #     print("⚠️ Content parts hoàn toàn rỗng.")

        # 5. Xử lý kết quả
        print("\n--- KẾT QUẢ ĐÁNH GIÁ TỪ GEMINI ---")
        
        try:
            json_str = clean_json_response(response_text)
            result = json.loads(json_str)
            
            # In đẹp kết quả JSON
            print(json.dumps(result, indent=4, ensure_ascii=False))
            
            # In tóm tắt thân thiện
            print("\n------------------------------------")
            def print_model_stats(name, data):
                total = data['accuracy_score'] + data['fluency_score'] + data['terminology_score']
                print(f"👉 [{name}] - Tổng điểm: {total}/30")
                print(f"   {'• Accuracy (Chính xác):':<30} {data['accuracy_score']}/10")
                print(f"   {'• Fluency (Trôi chảy):':<30} {data['fluency_score']}/10")
                print(f"   {'• Terminology (Thuật ngữ):':<30} {data['terminology_score']}/10")
                print(f"   💬 Nhận xét: {data['comment']}")

            print_model_stats("MODEL A", result.get('model_a', {}))
            print("\n------------------------------------")
            print_model_stats("MODEL B", result.get('model_b', {}))

            print(f"🏆 NGƯỜI CHIẾN THẮNG: {result.get('winner', 'Unknown')}")
            print(f"📝 LÝ DO: {result.get('reasoning', 'No reason provided')}")
            print("------------------------------------")

        except json.JSONDecodeError:
            print("⚠️ Không thể parse JSON từ phản hồi. Dưới đây là nội dung thô:")
            print(response_text)

    except Exception as e:
        print(f"\n❌ Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()
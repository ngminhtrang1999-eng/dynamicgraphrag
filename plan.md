Refactoring Strategy: Dynamic GraphRAG "Draft-then-Edit" Pipeline
1. Executive Strategy & Purpose
Mục tiêu: Tối ưu hóa sự đánh đổi giữa Độ trôi chảy (Fluency) và Tính nhất quán (Consistency) trong hệ thống dịch thuật tiểu thuyết sử dụng LLM nhỏ (7B-8B parameters).

Vấn đề hiện tại (Single-Pass Injection):

Việc chèn thẻ [Entity] trực tiếp vào văn bản gốc (He [Klein] looked...) làm gián đoạn luồng ngôn ngữ tự nhiên, khiến các model nhỏ bị "rối" (perplexity cao), dẫn đến câu văn sượng hoặc ảo giác (hallucination).

Kết quả: Điểm M-ETA (Entity Accuracy) cao nhưng BLEURT/COMET (Fluency) thấp hơn so với dịch thường.

Giải pháp chiến lược (Two-Pass Post-Editing): Chuyển đổi sang kiến trúc "Dịch trước - Sửa sau":

Pass 1 (Drafting): Dịch thuần túy (Clean Input) để tối đa hóa Fluency.

Pass 2 (Editing): Sử dụng Graph Context (Entity Injection + Glossary) làm Metadata tham chiếu để chỉnh sửa lại bản nháp, đảm bảo Consistency.

2. Architecture & Data Flow
Quy trình cũ (Deprecated for Logic):
Source + [Injection] -> LLM -> Translation

Quy trình mới (Target):
Step A (Drafting): Clean Source -> LLM (Creative Mode) -> Draft Translation

Step B (Context Prep): Clean Source -> Injection LLM -> Injected Text (Reference Only)

Step C (Editing): Source + Draft + Reference + Graph Rules -> LLM (Strict Mode) -> Final Translation

3. Implementation Plan (Step-by-Step)
Step 1: Implement "Drafting" Prompt (Phase 1)
Purpose: Tạo ra một bản dịch tự nhiên nhất có thể, không bị gò bó bởi các luật lệ kỹ thuật.

Input:

chunk_text: Văn bản gốc sạch sẽ.

prev_context: 1-2 câu cuối của chunk trước (để giữ mạch văn).

Prompt Logic: Role "Tiểu thuyết gia". Yêu cầu dịch thoát ý, trôi chảy. Tuyệt đối không đưa JSON/Graph vào đây.

Expected Output: Bản dịch tiếng Việt mượt mà (có thể sai đại từ nhân xưng, sai thuật ngữ).

Step 2: Implement "Look-ahead" Context Preparation
Purpose: Tận dụng lại logic Entity Injection hiện có nhưng đổi mục đích sử dụng. Thay vì sửa input để dịch, ta dùng nó làm "Bằng chứng" cho Editor.

Action:

Giữ nguyên hàm gọi injection_llm.

Input: chunk_text.

Output: injected_text (VD: "He [Klein] said to him [Dunn]").

Lưu ý: Biến này sẽ KHÔNG được đưa vào Prompt Drafting, mà được chuyển thẳng xuống bước Editing.

Step 3: Implement "Post-Editing" Prompt (Phase 2)
Purpose: Sửa lỗi bản nháp dựa trên bằng chứng và luật.

Input:

source_text: Văn bản gốc.

draft_text: Kết quả từ Step 1.

reference_text: Kết quả từ Step 2 (injected_text).

rule_set: Glossary và Relations đã được lọc (filtered) chỉ chứa các thực thể xuất hiện trong chunk.

Prompt Logic: Role "Biên tập viên".

Nhiệm vụ: So sánh Draft với Reference và Rules.

Constraint: CHỈ sửa chỗ sai (thuật ngữ, xưng hô). GIỮ NGUYÊN văn phong chỗ đúng.

Expected Output: Bản dịch hoàn chỉnh, văn phong mượt (từ Step 1) và đúng xưng hô (từ Step 3).

4. File Modifications Specification
A. prompts.py
Cần thêm 2 hàm mới (giữ lại các hàm cũ để backup/so sánh nếu cần).

build_draft_translation_prompt(chunk_text, prev_context)

Construct simple system prompt focused on literary translation style.

Append previous context to guide the flow.

build_post_editing_prompt(source_text, draft_text, injected_reference, glossary, relations)

Structure:

### RULE SET: Convert glossary dict & relation list into natural language bullet points.

### REFERENCE CONTEXT: Insert injected_reference.

### SOURCE: source_text.

### DRAFT: draft_text.

Instruction: Explicitly tell the model to look at REFERENCE to resolve pronouns in SOURCE, then fix DRAFT.

B. pipeline.py
Tạo class mới PostEditingPipeline(BasePipeline) (hoặc sửa GraphRAGPipeline nếu muốn thay thế hoàn toàn).

Logic trong process_chapter:

Loop qua các chunks.

Call 1: Gọi LLM với build_draft_translation_prompt. Lưu result vào draft_trans.

Logic Injection: Gọi injection_llm (hoặc reuse logic cũ) để lấy injected_chunk.

Logic Filter: Viết helper function để lọc self.gm.glossary và self.gm.relations -> Chỉ lấy items có key xuất hiện trong chunk_text.

Call 2: Gọi LLM với build_post_editing_prompt.

Save: Ghi kết quả cuối cùng ra file.

C. graph_manager.py (Optional but Recommended)
Thêm phương thức helper: get_relevant_context(chunk_text)

Input: Text đoạn hiện tại.

Logic: Quét text, tìm các từ khóa có trong Glossary và Node IDs.

Output: Subset Glossary và Subset Relations (giúp giảm token cho Prompt Editing).

D. main.py
Thêm argument --pipeline type (hoặc tự động chuyển logic dựa trên flag) để kích hoạt PostEditingPipeline.

5. Verification & Acceptance Criteria
Quy trình kiểm thử (Verification Process):
Test Hallucination:

Chạy với model nhỏ (Llama-3-8b/Gemini Flash).

Kiểm tra xem bản dịch cuối có bịa ra đoạn văn nào không có trong Source không (do Input Step 3 khá dài).

Fix: Nếu có, cần tối ưu lại bộ lọc Rule Set để ngắn gọn hơn.

Test Consistency (Quan trọng):

Chọn đoạn hội thoại giữa 2 nhân vật có quan hệ phức tạp (VD: Klein vs Captain).

So sánh Draft (chưa sửa) và Final (đã sửa).

Expectation: Draft có thể dịch là "Anh - Tôi", nhưng Final phải sửa thành "Đội trưởng - Tôi".

Test Fluency:

Đọc bản Final. Nó có mượt như bản dịch thuần (SimplePipeline) không?

Expectation: Phải mượt hơn bản GraphRAGPipeline cũ (do không bị chèn tag [] vào giữa câu).

Input/Output Example for Verification:
Input (Source):

He walked to the desk. "Is this 0-08?" he asked.

Intermediate (Draft):

Anh ấy đi tới bàn làm việc. "Đây có phải là 0-08 không?" anh ấy hỏi.

Intermediate (Reference):

He [Klein] walked to the desk. "Is this 0-08 [Artifact]?" he [Klein] asked.

Intermediate (Rule):

Klein -> Tự xưng "Tôi".

0-08 -> Vật Phong Ấn 0-08.

Expected Final Output:

Tôi đi tới bàn làm việc. "Đây có phải là Vật Phong Ấn 0-08 không?" tôi hỏi.
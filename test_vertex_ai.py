# import vertexai
# from vertexai.generative_models import GenerativeModel

# # 1. Khởi tạo kết nối (Thay thế bằng Project ID và Region của bạn)
# project_id = "gen-lang-client-0302949969"
# location = "us-central1"

# vertexai.init(project=project_id, location=location)

# # 2. Chọn Model (ví dụ: gemini-1.5-flash hoặc gemini-1.0-pro)
# model = GenerativeModel("qwen/qwen2@qwen2.5-7b-instruct")

# # 3. Gửi prompt và nhận phản hồi
# response = model.generate_content("Xin chào. Bạn là ai?")

# # 4. In kết quả
# print(response.text)

from anthropic import AnthropicVertex

LOCATION="us-east5"

client = AnthropicVertex(region=LOCATION, project_id="gen-lang-client-0302949969")

message = client.messages.create(
  max_tokens=1024,
  messages=[
    {
      "role": "user",
      "content": "Send me a recipe for banana bread.",
    }
  ],
  model="claude-3-5-haiku@20241022",
)
print(message.model_dump_json(indent=2))
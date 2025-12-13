
import const
import os
import time  # [NEW] Import time để dùng sleep
import requests
import json
import utils
from abc import ABC, abstractmethod
from openai import OpenAI
# [Existing imports...]


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, messages, temperature=0.3):
        pass

class LocalQwenLLM(BaseLLM):
    def __init__(self):
        print("⏳ ĐANG KHỞI TẠO LOCAL MODEL (4-bit Quantization)...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(const.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            const.MODEL_NAME, quantization_config=self.bnb_config, device_map="auto"
        )

    def generate(self, messages, temperature=0.3):
        import torch
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([inputs], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs, 
                max_new_tokens=const.MAX_NEW_TOKENS, 
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1
            )
        return self.tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

class GeminiAPILLM(BaseLLM):
    def __init__(self, model_name=None):
        # Ưu tiên model_name truyền vào, nếu không có thì lấy từ const
        target_model = model_name if model_name else const.GEMINI_MODEL_NAME
        
        print(f"⏳ ĐANG KẾT NỐI GEMINI API ({target_model})...")
        import google.generativeai as genai
        if not const.GEMINI_API_KEY:
            raise ValueError("⚠️ Chưa cấu hình GEMINI_API_KEY trong const.py")
        genai.configure(api_key=const.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(target_model)

    def generate(self, messages, temperature=0.3):
        gemini_history = []
        system_instruction = None

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = content 
            elif role == "assistant":
                gemini_history.append({"role": "model", "parts": [content]})
            else:
                gemini_history.append({"role": "user", "parts": [content]})

        chat = self.model.start_chat(history=gemini_history[:-1]) 
        
        generation_config = {"temperature": temperature, "max_output_tokens": const.MAX_NEW_TOKENS}
        
        last_msg = gemini_history[-1]["parts"][0]
        if system_instruction:
            last_msg = f"System Instruction: {system_instruction}\n\nTask: {last_msg}"

        response = chat.send_message(last_msg, safety_settings=safety_settings, generation_config=generation_config, request_options={"timeout": 600})
        return response.text
    
class GroqLLM(BaseLLM):
    def __init__(self, model_name="llama-3.1-8b-instant"):
        """
        Initializes Groq client. 
        Ensure GROQ_API_KEY is set in your environment variables or const.py
        """
        # Lazy import
        from groq import Groq

        target_model = model_name
        print(f"⏳ INITIALIZING GROQ API ({target_model})...")
                
        self.client = Groq(api_key=const.GROQ_API_KEY)
        self.model_name = target_model

    def generate(self, messages, temperature=0.3):
        # [NEW] Throttling Logic for Groq Free Tier
        # Đếm ngược 15 giây trước mỗi lần gọi API
        print(f"   ⏳ Rate Limit Guard: Sleeping 30s...", end="", flush=True)
        for i in range(30, 0, -1):
            time.sleep(1)
            # Sử dụng \r để ghi đè dòng hiện tại
            print(f"\r   ⏳ Rate Limit Guard: Sleeping {i}s...   ", end="", flush=True)
        
        # Xóa dòng đếm ngược và thông báo đang gửi request
        print(f"\r   🚀 Sending Request to Groq...                 ")

        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=temperature,
                max_tokens=const.MAX_NEW_TOKENS, # Ensure this exists in const.py
                top_p=1,
                stop=None,
                stream=False
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"   ⚠️ Groq API Error: {e}")
            raise e

class OpenRouterLLM(BaseLLM):
    def __init__(self, model_name="google/gemma-3-27b-it:free"):
        self.model_name = model_name
        self.api_key = const.OPENROUTER_API_KEY
        print(f"⏳ INITIALIZING OPENROUTER API ({self.model_name})...")

    def generate(self, messages, temperature=0.3):
        print(f"   🚀 Sending Request to OpenRouter ({self.model_name})...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # OpenRouter expects "messages" and "model"
        payload = {
            "model": self.model_name,
            "messages": messages,
            # "temperature": temperature, # Optional, adding if supported/needed
        }
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                response_json = response.json()
                content = utils.extract_api_response_content(response_json)
                if content:
                    return content
                else:
                    return "" # Handle empty or failed extraction
            else:
                print(f"   ⚠️ OpenRouter API Error: {response.status_code} - {response.text}")
                raise Exception(f"OpenRouter API Error: {response.text}")
                
        except Exception as e:
            print(f"   ⚠️ Request Failed: {e}")
            raise e

class LangDBLLM(BaseLLM):
    def __init__(self, model_name="deepinfra/gemma-3-27b-it"):
        self.model_name = model_name
        print(f"⏳ INITIALIZING LANGDB API ({self.model_name})...")        
        
        server_url = "https://api.us-east-1.langdb.ai"
        
        self.client = OpenAI(
            api_key=const.LANGDB_API_KEY, 
            base_url=server_url,
            default_headers={
                "x-project-id": const.LANGDB_PROJECT_ID
            }
        )

    def generate(self, messages, temperature=0.3):
        print(f"   🚀 Sending Request to LangDB ({self.model_name})...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=const.MAX_NEW_TOKENS
            )
            
            return response.choices[0].message.content

        except Exception as e:
            print(f"   ⚠️ LangDB API Error: {e}")
            raise e

class NVIDIALLM(BaseLLM):
    def __init__(self, model_name="qwen/qwen2.5-7b-instruct"):
        self.model_name = model_name
        print(f"⏳ INITIALIZING NVIDIA API ({self.model_name})...")
        
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=const.NVIDIA_API_KEY
        )

    def generate(self, messages, temperature=0.3):
        print(f"   ⏳ Rate Limit Guard: Sleeping 20s...", end="", flush=True)
        for i in range(20, 0, -1):
            time.sleep(1)
            # Sử dụng \r để ghi đè dòng hiện tại
            print(f"\r   ⏳ Rate Limit Guard: Sleeping {i}s...   ", end="", flush=True)
        print(f"\r   🚀 Sending Request to Nvidia ({self.model_name})...")  
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=0.7,
                max_tokens=const.MAX_NEW_TOKENS,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"   ⚠️ Nvidia API Error: {e}")
            raise e

class CloudfareLLM(BaseLLM):
    def __init__(self, model_name="@cf/meta/llama-3-8b-instruct"):
        self.model_name = model_name
        self.account_id = const.CLOUDFARE_ACCOUNT_ID
        self.api_token = const.CLOUDFARE_API_KEY
        print(f"⏳ INITIALIZING CLOUDFARE API ({self.model_name})...")
        
        if not self.account_id or not self.api_token:
             raise ValueError("⚠️ Config missing for Cloudflare (ACCOUNT_ID or API_KEY)")

    def generate(self, messages, temperature=0.3):
        print(f"   ⏳ Rate Limit Guard: Sleeping 20s...", end="", flush=True)
        for i in range(20, 0, -1):
            time.sleep(1)
            # Sử dụng \r để ghi đè dòng hiện tại
            print(f"\r   ⏳ Rate Limit Guard: Sleeping {i}s...   ", end="", flush=True)
        print(f"\r   🚀 Sending Request to Cloudflare ({self.model_name})...")
    
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Cloudflare implementation in test_cloudfare_api.py uses "role": "system" correctly
        payload = {
            "messages": messages
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result["result"]["response"]
            else:
                print(f"   ⚠️ Cloudflare API Error: {response.text}")
                raise Exception(f"Cloudflare API Error: {response.text}")
        except Exception as e:
            print(f"   ⚠️ Request Failed: {e}")
            raise e
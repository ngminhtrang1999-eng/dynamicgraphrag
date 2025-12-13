import requests

# 1. SETUP CREDENTIALS
ACCOUNT_ID = const.CLOUDFARE_ACCOUNT_ID
API_TOKEN = const.CLOUDFARE_API_KEY

# 2. CHOOSE MODEL
model_id = "@cf/meta/llama-3-8b-instruct"

# 3. CONSTRUCT ENDPOINT
url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{model_id}"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# 4. PREPARE INPUT (Standard chat format)
payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in one sentence."}
    ]
}

# 5. CALL API
response = requests.post(url, headers=headers, json=payload)
result = response.json()

if response.status_code == 200:
    print("Response:", result["result"]["response"])
else:
    print("Error:", response.text)
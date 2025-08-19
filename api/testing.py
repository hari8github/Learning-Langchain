import os, requests
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
r = requests.post(
  "https://api.groq.com/openai/v1/chat/completions",
  headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
  json={
    "model": "llama3-8b-8192",  # use your actual model, or one Groq supports
    "messages": [{"role":"user","content":"Hello"}]
  }
)
print(r.status_code)
print(r.text)
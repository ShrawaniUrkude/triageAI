import ollama
from decouple import config

response = ollama.chat(
    model=config("OLLAMA_MODEL", default="mistral:7b"),
    messages=[
        {
            "role": "user",
            "content": "Give me 5 names of animals",
        }
    ],
)
print(response.message.content)

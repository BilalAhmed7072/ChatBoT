import requests
import json

url = "http://127.0.0.1:5000/chatbot"
data = {"prompt": "Hello, how are you?"}

response = requests.post(url, json=data)
print(response.text)

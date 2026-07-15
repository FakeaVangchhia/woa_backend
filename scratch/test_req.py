import requests

url = "http://127.0.0.1:8420/agent/user-xOs7DZVX/run"
payload = {
    "provider": "gemini",
    "model": "gemini-invalid-model-name-xyz",
    "api_key": "some-key",
    "goal": "hello",
    "ticks": 1
}
headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("Error:", e)

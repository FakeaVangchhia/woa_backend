import requests

url = "http://127.0.0.1:8420/agent/user-xOs7DZVX/run"
payload = {
    "provider": "gemini",
    "model": "gemini-1.5-flash",
    "api_key": "dummy-key-for-test",
    "goal": "Verify server is online",
    "ticks": 1
}

print("Testing connection to local backend on port 8420...")
try:
    res = requests.post(url, json=payload, timeout=5)
    print("Status Code:", res.status_code)
    print("Response JSON:", res.json())
    if res.status_code == 400 and "API key not valid" in res.text:
        print("SUCCESS: Route is active and responding correctly! (400 is expected for dummy API key)")
    elif res.status_code == 404:
        print("ERROR: Route returned 404. You may have a stale server process running on port 8420.")
    else:
        print("Response received:", res.status_code)
except Exception as e:
    print("CONNECTION ERROR: Is uvicorn running?", e)

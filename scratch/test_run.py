import urllib.request
import urllib.error
import json

url = "http://127.0.0.1:8420/agent/test-agent-123/run"
headers = {"Content-Type": "application/json"}
body = {
    "provider": "gemini",
    "model": "gemini-1.5-flash",
    "api_key": "invalid-key-xyz",
    "goal": "Build a small house",
    "ticks": 1
}

req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
try:
    with urllib.request.urlopen(req) as res:
        print("Response Code:", res.status)
        print("Response Body:", res.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    print("HTTPError Code:", e.code)
    print("HTTPError Response:", e.read().decode("utf-8"))
except Exception as e:
    print("Error:", e)

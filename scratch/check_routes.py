import requests

try:
    res = requests.get("http://127.0.0.1:8420/openapi.json")
    if res.ok:
        data = res.json()
        print("Paths registered on the running server:")
        for path in sorted(data.get("paths", {}).keys()):
            print(f"  {path} -> {list(data['paths'][path].keys())}")
    else:
        print("Failed to fetch openapi.json, status:", res.status_code)
except Exception as e:
    print("Error:", e)

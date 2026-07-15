from app.main import app

print("Registered Routes:")
for route in app.routes:
    # Print route path and methods
    methods = getattr(route, "methods", None)
    print(f"Path: {route.path} | Methods: {methods} | Name: {route.name}")

import asyncio
import websockets
import json

async def test_connect():
    uri = "ws://127.0.0.1:8420/ws"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected successfully!")
            # Start listening for messages
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"Received message: {message[:100]}...")
                except asyncio.TimeoutError:
                    print("No message received for 5 seconds (heartbeat/idle check)")
                    # Send a simple ping to see if connection is alive
                    # In this app, the backend's receive_text expects messages
                    await websocket.send("ping")
                    print("Sent ping/heartbeat to keep connection alive.")
    except Exception as e:
        print(f"WebSocket error: {e}")

asyncio.run(test_connect())

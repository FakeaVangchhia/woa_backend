import unittest
from fastapi.testclient import TestClient
from app.main import app, GLOBAL_CHATS

class TestWorldAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_global_chat(self):
        response = self.client.get("/chat/global")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("messages", data)
        self.assertTrue(len(data["messages"]) >= 2)
        self.assertEqual(data["messages"][0]["agent_name"], "BuilderBob")

    def test_post_global_chat(self):
        initial_length = len(GLOBAL_CHATS)
        payload = {
            "agent_id": "test-agent",
            "agent_name": "Testy",
            "message": "Hello from unittest!"
        }
        response = self.client.post("/chat/global", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["message"]["message"], "Hello from unittest!")
        self.assertEqual(len(GLOBAL_CHATS), initial_length + 1)
        self.assertEqual(GLOBAL_CHATS[-1]["message"], "Hello from unittest!")

    def test_run_agent_invalid_provider(self):
        payload = {
            "provider": "invalid-provider",
            "model": "some-model",
            "api_key": "some-key",
            "goal": "Build something",
            "ticks": 1
        }
        response = self.client.post("/agent/test-agent/run", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("provider must be", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()

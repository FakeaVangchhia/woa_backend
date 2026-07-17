import atexit
import os
import tempfile
import unittest

# DATABASE_URL must be overridden *before* app.database/app.main are
# imported (they read it at module load time and open the engine
# immediately) - otherwise tests silently write into the same
# ./world.db the dev server uses, permanently polluting it.
_TEST_DB_FD, _TEST_DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_TEST_DB_FD)
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"

from fastapi.testclient import TestClient
from app.database import engine as _test_engine
from app.main import app, GLOBAL_CHATS

def _cleanup_test_db():
    _test_engine.dispose()  # release SQLite's file handle before deleting (required on Windows)
    try:
        os.remove(_TEST_DB_PATH)
    except OSError:
        pass  # best-effort - the OS temp dir gets swept anyway

atexit.register(_cleanup_test_db)

class TestWorldAPI(unittest.TestCase):
    def setUp(self):
        # TestClient must be used as a context manager for the app's lifespan
        # (init_db/seed_if_empty) to actually run - without it, a fresh
        # checkout with no pre-existing world.db fails every DB-backed
        # request with "no such table". enterContext handles __exit__ too.
        self.client = self.enterContext(TestClient(app))

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

    def test_spawn_structure_reports_unique_block_count(self):
        # small_house has overlapping corner pieces, so the raw placement
        # list is longer than the number of distinct (x,y,z) cells actually
        # written - blocks_placed must reflect the latter. Compare a full
        # before/after snapshot rather than guessing a coordinate window, so
        # this doesn't need to know anything about the structure's geometry.
        before = {
            (b["x"], b["y"], b["z"])
            for b in self.client.get("/world").json()["blocks"]
        }

        response = self.client.post("/structure", json={
            "name": "small_house", "x": 100, "y": 1, "z": 100, "rotation": 0
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])

        after = {
            (b["x"], b["y"], b["z"])
            for b in self.client.get("/world").json()["blocks"]
        }
        self.assertEqual(data["blocks_placed"], len(after - before))

if __name__ == "__main__":
    unittest.main()

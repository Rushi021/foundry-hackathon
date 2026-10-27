from fastapi.testclient import TestClient

from foundry import api
from foundry.api import app


def test_health():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_run_passes_session_and_returns_status(monkeypatch):
    seen = {}

    def fake_run(prompt, *, session_id=None, verbose=True):
        seen.update(prompt=prompt, session_id=session_id)
        return {
            "run_id": "r",
            "session_id": session_id or "new",
            "status": "completed",
            "route": "direct",
            "route_rationale": "router",
            "route_confidence": 0.9,
            "decision": None,
            "response": "hi",
            "task": {},
            "metrics": {},
            "latency_ms": 1,
        }

    monkeypatch.setattr(api, "run_workflow", fake_run)
    res = TestClient(app).post("/v1/workflows/run", json={"prompt": "hey", "session_id": "s1"})
    assert res.status_code == 200
    assert res.json()["status"] == "completed"
    assert seen == {"prompt": "hey", "session_id": "s1"}


def test_memory_endpoints(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api, "get_settings", lambda: type("S", (), {"foundry_data_dir": str(tmp_path)})()
    )
    client = TestClient(app)
    assert client.post("/v1/memory", json={"scope": "private", "content": "x"}).status_code == 200
    assert [m["content"] for m in client.get("/v1/memory/private").json()] == ["x"]
    assert client.get("/v1/memory/shared").json() == []
    assert client.post("/v1/memory", json={"scope": "secret", "content": "x"}).status_code == 422

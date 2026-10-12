import pytest

from agents import graph
from lib.db import get_conn, init_db


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr("lib.db.DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setattr("lib.trace.PATH", str(tmp_path / "trace.jsonl"))
    init_db()
    conn = get_conn()
    conn.execute("INSERT INTO restaurants VALUES (1,'A Cafe','Cafe','X',4.5)")
    conn.commit()
    conn.close()


def test_unknown_restaurant_rejected(db):
    with pytest.raises(ValueError):
        graph.run(999)


def test_worker_crash_does_not_kill_run(db, monkeypatch):
    class Decision:
        def __init__(self, next, reason="", instruction=""):
            self.next, self.reason, self.instruction = next, reason, instruction

    calls = {"n": 0}

    def fake_decide(rid, findings, steps):
        calls["n"] += 1
        return Decision("monitor" if calls["n"] == 1 else "finish", "test")

    def boom(rid, instruction=""):
        raise RuntimeError("tool exploded")

    monkeypatch.setattr(graph.supervisor, "decide", fake_decide)
    monkeypatch.setattr(graph.monitor, "run", boom)
    out = graph.run(1)
    assert "RuntimeError" in out["findings"]["monitor"][0]["error"]


def test_supervisor_failure_finishes_cleanly(db, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(graph.supervisor, "decide", fail)
    monkeypatch.setattr(graph.supervisor, "Decision", lambda **k: type("D", (), {"instruction": "", **k})())
    out = graph.run(1)
    assert out["findings"] == {}

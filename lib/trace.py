import json
import uuid
from datetime import datetime, timezone

PATH = "trace.jsonl"
_run = {"id": None}


def new_run():
    _run["id"] = uuid.uuid4().hex[:8]
    return _run["id"]


def log(sender, receiver, content, reason=""):
    rec = {
        "run_id": _run["id"],
        "ts": datetime.now(timezone.utc).isoformat(),
        "from": sender,
        "to": receiver,
        "reason": reason,
        "content": content,
    }
    with open(PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(f"[{sender} -> {receiver}] {reason}")
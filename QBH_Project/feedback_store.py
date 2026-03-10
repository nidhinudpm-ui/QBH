import json
import os
import threading
from datetime import datetime

FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "feedback_log.json")
_lock = threading.Lock()

def log_feedback(entry):
    """
    Append a feedback entry to the JSON log file.
    Entry should be a dict.
    """
    with _lock:
        data = []
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        
        entry["timestamp"] = datetime.now().isoformat()
        data.append(entry)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

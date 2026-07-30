#!/usr/bin/env python
"""A-Insight notification — standalone script for GitHub Actions."""
import os, sys, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def send_server_chan(key, title, text):
    import requests
    url = f"https://sctapi.ftqq.com/{key}.send"
    try:
        resp = requests.post(url, data={"title": title, "desp": text}, timeout=10)
        result = resp.json()
        print(f"  Response: {result}")
        return result.get("code") == 0
    except Exception as e:
        print(f"  Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 40)
    print("Notify Test V3 — NEW SCRIPT")
    print("=" * 40)

    key = os.environ.get("SERVER_CHAN_KEY", "")
    smtp_host = os.environ.get("SMTP_HOST", "")

    print(f"server_chan_key: {'SET' if key else 'MISSING'}")
    print(f"smtp_host: {smtp_host or 'MISSING'}")

    if key:
        print("Sending Server酱 push...")
        title = f"A-Insight {datetime.now().strftime('%Y-%m-%d')}"
        ok = send_server_chan(key, title, "Test push from GitHub Actions")
        if ok:
            print("SUCCESS: WeChat push sent!")
        else:
            print("FAILED: Server酱 push failed")
    else:
        print("No SERVER_CHAN_KEY configured, skipping push")

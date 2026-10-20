"""Start the intrusion-detection server. It asks for the camera token first.

    python run_server.py
"""
import os
import subprocess

import uvicorn

os.environ.setdefault("CAMERA_URL", "http://172.20.10.3/capture")
print(f"Camera address: {os.environ['CAMERA_URL']}")
raw = input("Paste the camera token and press Enter: ")
# drop hidden/non-ASCII characters (BOM, zero-width, curly quotes) that sneak in when pasting
os.environ["CAMERA_TOKEN"] = "".join(c for c in raw if 32 < ord(c) < 127 and c not in "\"'")

# free port 8000 if an old server is still running there (not this process)
subprocess.run(
    ["powershell", "-NoProfile", "-Command",
     f"Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | "
     f"Where-Object {{ $_.OwningProcess -ne {os.getpid()} }} | "
     f"ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force }}"],
    check=False,
)

print("\nStarting server... wait about 20 seconds for 'Application startup complete'.")
print("Keep this window open. Press Ctrl+C to stop.\n")
uvicorn.run("server:app", host="0.0.0.0", port=8000)

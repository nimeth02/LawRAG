"""Start the server:  python -m app

Host and port come from APP_HOST / APP_PORT in .env (defaults 127.0.0.1:8080).
Port 8080 is the default because 8000 is commonly taken by other local servers.
"""
from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    host = os.getenv("APP_HOST") or "127.0.0.1"
    port = int(os.getenv("APP_PORT") or 8080)
    reload = (os.getenv("APP_RELOAD") or "true").lower() in ("1", "true", "yes")
    print(f"LankaLaw UI: http://{host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()

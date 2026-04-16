"""
app.py
------
Entry point for the Financial Research Assistant.

Starts the FastAPI server that serves both the REST/SSE API and the
built React frontend from `frontend/dist`.

For local development:
    # Terminal 1: API server
    python app.py

    # Terminal 2: React dev server (hot reload)
    cd frontend && npm run dev
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    import uvicorn
    from src.config import settings

    dist = Path(__file__).parent / "frontend" / "dist"
    if not dist.exists():
        print("Frontend not built yet. Run:")
        print("  cd frontend && npm install && npm run build")
        print()
        print("Or for development with hot reload:")
        print("  cd frontend && npm run dev")
        print()
        print("Starting API server anyway (API available at /api/ask) ...")
        print()

    uvicorn.run(
        "src.app.server:app",
        host="0.0.0.0",
        port=settings.APP_PORT,
        reload=False,
    )

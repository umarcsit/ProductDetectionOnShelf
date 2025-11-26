import sys
from pathlib import Path

# Add parent directory to path so 'app' package can be imported
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn

try:
    from app.core.config import settings
    from app.api.routes import router as api_router
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).parent}")
    print(f"Parent directory: {parent_dir}")
    print(f"Python path: {sys.path}")
    raise

app = FastAPI(title=settings.PROJECT_NAME)

# API routes
app.include_router(api_router, prefix="/api")

# Static frontend (only mount if directory exists)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/index.html")


if __name__ == "__main__":
    try:
        print("=" * 50)
        print("Starting Visual Search API Server...")
        print("=" * 50)
        print(f"Server will be available at http://localhost:8000")
        print(f"API docs at http://localhost:8000/docs")
        print(f"Health check at http://localhost:8000/api/health")
        print("Press Ctrl+C to stop the server\n")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,  # Set to True if you want auto-reload (may cause issues on Windows)
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\n\nERROR: Failed to start server")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 50)
        input("Press Enter to exit...")

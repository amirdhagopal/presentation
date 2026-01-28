#!/usr/bin/env python3
"""
Unified launcher for AI Presentation Modules.
Usage: python run.py [module_name]
Example: python run.py agents
"""

import sys
import importlib
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

def create_app(module_name: str) -> FastAPI:
    """Create a FastAPI app for the specified module."""
    base_path = Path(__file__).parent / "modules" / module_name
    
    if not base_path.exists():
        print(f"Error: Module '{module_name}' not found at {base_path}")
        sys.exit(1)

    app = FastAPI(title=f"{module_name.capitalize()} Presentation")

    # 1. Mount Slides (if exists)
    slides_path = base_path / "slides"
    if slides_path.exists():
        app.mount("/slides", StaticFiles(directory=slides_path), name="slides")
        print(f"Mounted slides from: {slides_path}")

    # 2. Mount Demo Static Files (if exists)
    demo_static_path = base_path / "demo" / "static"
    if demo_static_path.exists():
        app.mount("/demo/static", StaticFiles(directory=demo_static_path), name="demo_static")
        print(f"Mounted demo static from: {demo_static_path}")

    # 3. Include Demo API Router (if api.py exists)
    api_module_path = f"modules.{module_name}.demo.api"
    try:
        api_module = importlib.import_module(api_module_path)
        if hasattr(api_module, "router"):
            app.include_router(api_module.router)
            print(f"Included API router from: {api_module_path}")
    except ModuleNotFoundError:
        print(f"No API module found at {api_module_path} (skipping)")
    except Exception as e:
        print(f"Error loading API module: {e}")

    # Print Access URLs
    print("\n" + "="*50)
    print(f"🚀 {module_name.capitalize()} Presentation Running")
    print("="*50)
    if slides_path.exists():
        print(f"📝 Slides: http://localhost:8001/slides/index.html")
    if demo_static_path.exists():
        print(f"🎮 Demo:   http://localhost:8001/demo/static/index.html")
    print(f"🔌 API:    http://localhost:8001/docs")
    print("="*50 + "\n")

    # 4. Root Redirect
    @app.get("/")
    async def root():
        if slides_path.exists():
            return RedirectResponse(url="/slides/index.html")
        return {"message": f"Welcome to {module_name} presentation"}

    # 5. Demo Redirect (convenience)
    @app.get("/demo")
    async def demo_redirect():
        return RedirectResponse(url="/demo/static/index.html")

    return app

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py [module_name]")
        print("Example: python run.py ai_intro")
        print("Available modules:")
        # List directories in modules/
        modules_dir = Path(__file__).parent / "modules"
        if modules_dir.exists():
            for p in modules_dir.iterdir():
                if p.is_dir() and not p.name.startswith("__"):
                    print(f" - {p.name}")
        sys.exit(1)

    module = sys.argv[1]
    app = create_app(module)
    uvicorn.run(app, host="0.0.0.0", port=8001)

#!/usr/bin/env python3
"""
ChatWhiz FastAPI Server Launcher
Simple script to start the FastAPI backend server
"""

import uvicorn
import os
import sys
from pathlib import Path

def main():
    """Start the ChatWhiz FastAPI server"""
    
    # Add the project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    port = 8000
    print("Starting ChatWhiz FastAPI Server...")
    print(f"Project root: {project_root}")
    print(f"Server will be available at: http://localhost:{port}")
    print(f"API documentation at: http://localhost:{port}/docs")
    print(f"Web interface at: http://localhost:{port}/")
    print("-" * 50)
    
    # Start the FastAPI server
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

if __name__ == "__main__":
    main()

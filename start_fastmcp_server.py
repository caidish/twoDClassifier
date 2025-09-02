#!/usr/bin/env python3
"""
FastMCP server launcher with HTTP transport and port fallback
Starts the FastMCP server with HTTP transport on first available port
"""

import socket
import subprocess
import sys
from pathlib import Path


def is_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('localhost', port))
            return True
    except (socket.error, OSError):
        return False


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find the first available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")


def start_fastmcp_server(start_port: int = 8000):
    """Start the FastMCP server with HTTP transport and automatic port selection."""
    current_dir = Path(__file__).parent
    server_script = current_dir / "mcp_fastmcp_server.py"
    
    if not server_script.exists():
        print(f"Error: FastMCP server script not found at {server_script}")
        sys.exit(1)
    
    try:
        # Find available port
        port = find_available_port(start_port)
        print(f"Starting FastMCP server with HTTP transport on localhost:{port}")
        print(f"Server will be accessible at: http://localhost:{port}/mcp/")
        
        # Start the FastMCP server with HTTP transport
        cmd = [sys.executable, str(server_script), "--host", "localhost", "--port", str(port), "--transport", "http"]
        print(f"Executing: {' '.join(cmd)}")
        
        # Run the server
        process = subprocess.run(cmd)
        
        return process.returncode
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        return 0
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = start_fastmcp_server()
    sys.exit(exit_code)
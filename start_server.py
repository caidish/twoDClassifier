#!/usr/bin/env python3
"""
Unified MCP server launcher with port fallback
Starts either FastMCP (stdio) or HTTP MCP server with automatic port selection
"""

import argparse
import socket
import subprocess
import sys
import time
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
        return 1
    
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
        
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        return 0


def start_http_server(start_port: int = 8000):
    """Start the HTTP MCP server with automatic port selection."""
    current_dir = Path(__file__).parent
    server_script = current_dir / "mcp_http_server.py"
    
    if not server_script.exists():
        print(f"Error: HTTP server script not found at {server_script}")
        return 1
    
    try:
        # Find available port
        port = find_available_port(start_port)
        print(f"Starting HTTP MCP server on localhost:{port}")
        
        # Start the HTTP server
        cmd = [sys.executable, str(server_script), "--host", "localhost", "--port", str(port)]
        print(f"Executing: {' '.join(cmd)}")
        
        # Run the server
        process = subprocess.run(cmd)
        return process.returncode
        
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Start MCP server with automatic port selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Server Types:
  fastmcp: FastMCP server with HTTP transport (default)
  http:    Traditional HTTP/REST server for web clients and API access

Examples:
  python start_server.py                    # Start FastMCP HTTP server on first available port from 8000
  python start_server.py --type fastmcp     # Start FastMCP HTTP server (default)
  python start_server.py --port 9000        # Start FastMCP HTTP server from port 9000
  python start_server.py --type http --port 8001  # Start traditional HTTP server from port 8001
        """
    )
    
    parser.add_argument(
        "--type", 
        choices=["http", "fastmcp"], 
        default="fastmcp",
        help="Server type to start (default: fastmcp)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Starting port number for HTTP server (default: 8000)"
    )
    
    args = parser.parse_args()
    
    print(f"2D Material Classifier - MCP Server Launcher")
    print(f"Server type: {args.type}")
    
    if args.type == "fastmcp":
        return start_fastmcp_server(args.port)
    elif args.type == "http":
        return start_http_server(args.port)
    else:
        print(f"Unknown server type: {args.type}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
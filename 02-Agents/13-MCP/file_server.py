"""
File Operations MCP Server - provides file system operations
Run this as: python file_server.py
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("File Operations Server")

# Define safe directory for operations (sandbox)
SAFE_DIR = Path("./mcp_sandbox")
SAFE_DIR.mkdir(exist_ok=True)

def _get_safe_path(filename: str) -> Path:
    """Get a safe path within the sandbox directory."""
    safe_path = SAFE_DIR / filename
    # Ensure the path is within the safe directory
    if not str(safe_path.resolve()).startswith(str(SAFE_DIR.resolve())):
        raise ValueError(f"Access denied: {filename} is outside safe directory")
    return safe_path

@mcp.tool()
def create_file(filename: str, content: str = "") -> str:
    """Create a new file with optional content."""
    try:
        file_path = _get_safe_path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"✅ File '{filename}' created successfully with {len(content)} characters."
    except Exception as e:
        return f"❌ Error creating file '{filename}': {str(e)}"

@mcp.tool()
def read_file(filename: str) -> str:
    """Read the contents of a file."""
    try:
        file_path = _get_safe_path(filename)
        
        if not file_path.exists():
            return f"❌ File '{filename}' does not exist."
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"📖 Content of '{filename}':\n{content}"
    except Exception as e:
        return f"❌ Error reading file '{filename}': {str(e)}"

@mcp.tool()
def write_file(filename: str, content: str) -> str:
    """Write content to a file (overwrites existing content)."""
    try:
        file_path = _get_safe_path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"✅ File '{filename}' updated with {len(content)} characters."
    except Exception as e:
        return f"❌ Error writing to file '{filename}': {str(e)}"

@mcp.tool()
def append_to_file(filename: str, content: str) -> str:
    """Append content to an existing file."""
    try:
        file_path = _get_safe_path(filename)
        
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(content)
        
        return f"✅ Content appended to '{filename}'. Added {len(content)} characters."
    except Exception as e:
        return f"❌ Error appending to file '{filename}': {str(e)}"

@mcp.tool()
def list_files(directory: str = ".") -> str:
    """List files and directories in the specified directory."""
    try:
        dir_path = _get_safe_path(directory)
        
        if not dir_path.exists():
            return f"❌ Directory '{directory}' does not exist."
        
        if not dir_path.is_dir():
            return f"❌ '{directory}' is not a directory."
        
        items = []
        for item in sorted(dir_path.iterdir()):
            relative_path = item.relative_to(SAFE_DIR)
            if item.is_dir():
                items.append(f"📁 {relative_path}/")
            else:
                size = item.stat().st_size
                items.append(f"📄 {relative_path} ({size} bytes)")
        
        if not items:
            return f"📂 Directory '{directory}' is empty."
        
        return f"📂 Contents of '{directory}':\n" + "\n".join(items)
    except Exception as e:
        return f"❌ Error listing directory '{directory}': {str(e)}"

@mcp.tool()
def delete_file(filename: str) -> str:
    """Delete a file."""
    try:
        file_path = _get_safe_path(filename)
        
        if not file_path.exists():
            return f"❌ File '{filename}' does not exist."
        
        if file_path.is_dir():
            return f"❌ '{filename}' is a directory. Use delete_directory instead."
        
        file_path.unlink()
        return f"🗑️ File '{filename}' deleted successfully."
    except Exception as e:
        return f"❌ Error deleting file '{filename}': {str(e)}"

@mcp.tool()
def create_directory(directory: str) -> str:
    """Create a new directory."""
    try:
        dir_path = _get_safe_path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return f"📁 Directory '{directory}' created successfully."
    except Exception as e:
        return f"❌ Error creating directory '{directory}': {str(e)}"

@mcp.tool()
def delete_directory(directory: str) -> str:
    """Delete a directory and all its contents."""
    try:
        dir_path = _get_safe_path(directory)
        
        if not dir_path.exists():
            return f"❌ Directory '{directory}' does not exist."
        
        if not dir_path.is_dir():
            return f"❌ '{directory}' is not a directory."
        
        shutil.rmtree(dir_path)
        return f"🗑️ Directory '{directory}' and all its contents deleted successfully."
    except Exception as e:
        return f"❌ Error deleting directory '{directory}': {str(e)}"

@mcp.tool()
def file_info(filename: str) -> str:
    """Get detailed information about a file."""
    try:
        file_path = _get_safe_path(filename)
        
        if not file_path.exists():
            return f"❌ File '{filename}' does not exist."
        
        stat = file_path.stat()
        info = {
            "name": file_path.name,
            "path": str(file_path.relative_to(SAFE_DIR)),
            "size": stat.st_size,
            "is_directory": file_path.is_dir(),
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
        }
        
        # Format the information nicely
        result = f"ℹ️ Information for '{filename}':\n"
        result += f"  Name: {info['name']}\n"
        result += f"  Path: {info['path']}\n"
        result += f"  Size: {info['size']} bytes\n"
        result += f"  Type: {'Directory' if info['is_directory'] else 'File'}\n"
        result += f"  Created: {info['created']}\n"
        result += f"  Modified: {info['modified']}"
        
        return result
    except Exception as e:
        return f"❌ Error getting info for '{filename}': {str(e)}"

@mcp.tool()
def search_files(pattern: str, directory: str = ".") -> str:
    """Search for files matching a pattern in the specified directory."""
    try:
        dir_path = _get_safe_path(directory)
        
        if not dir_path.exists() or not dir_path.is_dir():
            return f"❌ Directory '{directory}' does not exist or is not a directory."
        
        matches = []
        for file_path in dir_path.rglob(pattern):
            if file_path.is_file():
                relative_path = file_path.relative_to(SAFE_DIR)
                size = file_path.stat().st_size
                matches.append(f"📄 {relative_path} ({size} bytes)")
        
        if not matches:
            return f"🔍 No files found matching pattern '{pattern}' in '{directory}'."
        
        return f"🔍 Found {len(matches)} file(s) matching '{pattern}':\n" + "\n".join(matches)
    except Exception as e:
        return f"❌ Error searching for files: {str(e)}"

if __name__ == "__main__":
    print(f"📁 Starting File Operations MCP Server...")
    print(f"🔒 Safe directory: {SAFE_DIR.absolute()}")
    mcp.run(transport="stdio")
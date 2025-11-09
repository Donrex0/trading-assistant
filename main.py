# main.py

import os
import subprocess
import sys

# Get project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Add project root to Python path
sys.path.insert(0, project_root)

# Run Streamlit with auto-reload enabled
streamlit_command = [
    "streamlit",
    "run",
    os.path.join("app", "dashboard.py"),
    "--server.port",
    "8502",  # Changed port to avoid conflicts
    "--server.address",
    "127.0.0.1",  # Use localhost instead of 0.0.0.0
    "--server.enableCORS",
    "false",
    "--server.enableXsrfProtection",
    "false",
    "--server.maxUploadSize",
    "1000",  # Increase upload size limit
    "--server.maxMessageSize",
    "1000",  # Increase message size limit
    "--server.baseUrlPath",
    ""  # Clear any base URL path
]

print(f"Starting Streamlit server from {project_root}")
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")
print(f"Streamlit command: {streamlit_command}")

try:
    subprocess.run(streamlit_command, check=True, capture_output=True, text=True)
except subprocess.CalledProcessError as e:
    print(f"Streamlit server failed: {e}")
    print(f"Output: {e.output}")
    print(f"Error: {e.stderr}")
    sys.exit(1)
subprocess.run(["streamlit", "run", "app/dashboard.py", "--server.fileWatcherType", "watchdog"], check=True)

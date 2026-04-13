"""
app.py
------
Entry point for HuggingFace Spaces deployment.
HuggingFace Spaces looks for app.py in the repo root.
This file simply imports and launches the Gradio app.

For local development, run:
    python app.py
    # or
    python -m src.app.gradio_app
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.app.gradio_app import build_ui

demo = build_ui()

# launch() with no arguments uses HuggingFace Spaces defaults
# (server_name and port are set by the Spaces environment)
if __name__ == "__main__":
    demo.launch()

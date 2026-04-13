"""
scripts/download_model.py
--------------------------
Downloads the recommended free GGUF LLM from HuggingFace Hub.
Run this once before starting the app.

Usage:
    python scripts/download_model.py

The model (Mistral-7B Q4_K_M) is ~4 GB and runs well on M-series Macs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import hf_hub_download  # official HF download utility

# ── Model config ──────────────────────────────────────────────────────────────
REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # 4-bit quantised, ~4 GB
MODELS_DIR = Path("./models")


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    dest = MODELS_DIR / FILENAME
    if dest.exists():
        print(f"✅ Model already downloaded at {dest}")
        return

    print(f"Downloading {FILENAME} (~4 GB) from HuggingFace Hub …")
    print("This will take a few minutes depending on your connection speed.")

    # hf_hub_download streams the file to local_dir
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,  # copy the file, don't symlink
    )

    print(f"✅ Model saved to {path}")
    print("Update LLM_MODEL_PATH in your .env if needed.")


if __name__ == "__main__":
    main()

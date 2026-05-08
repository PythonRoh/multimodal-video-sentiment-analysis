import requests
import tarfile
from pathlib import Path
from tqdm import tqdm  # nice lightweight progress bar

url = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
out_file = Path("MELD.Raw.tar.gz")
extract_dir = Path("data/MELD")

# Ensure output directory exists
extract_dir.parent.mkdir(parents=True, exist_ok=True)

print("⬇️ Downloading MELD dataset...")

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    total_size = int(r.headers.get("Content-Length", 0))
    chunk_size = 8192
    with open(out_file, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading"
    ) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

print("✅ Download complete:", out_file)

print("📦 Extracting...")
with tarfile.open(out_file, "r:gz") as tar:
    tar.extractall(path=extract_dir)

print("✅ Extraction done! Dataset available at:", extract_dir.resolve())

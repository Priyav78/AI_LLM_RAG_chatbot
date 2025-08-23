from pathlib import Path

# repo root (src is at: <repo>/Complete RAG folder/src/)
ROOT = Path(__file__).resolve().parents[2]
CRF  = ROOT / "Complete RAG folder"

DATA = CRF / "data"
RAW  = DATA / "raw"
PROC = DATA / "processed"

DOCS_LIST = RAW / "docs_list.txt"
DOCS_DIR  = RAW
EMBED     = PROC / "embed.npy"
INDEX     = PROC / "faiss_doc_index.index"

RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

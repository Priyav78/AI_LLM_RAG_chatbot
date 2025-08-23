# RAG Chatbot — File‑Grounded Question Answering

A compact Retrieval‑Augmented Generation (RAG) system that turns local documents into searchable embeddings (Sentence‑Transformers), indexes them with FAISS, retrieves the most relevant passages at query time, and generates grounded answers with an LLM. A lightweight Gradio UI is included for interactive testing and demos.

> **Why it matters**: Demonstrates an end‑to‑end, production‑style RAG pipeline with clean project structure, portable paths, and reproducible artifacts. Emphasis on correctness (cosine‑equivalent retrieval) and maintainability (no hard‑coded paths or keys).

---

## Key capabilities
- **Document ingestion** → read filenames from `data/raw/docs_list.txt`, load files from `data/raw/`, and create normalized embeddings with `all‑MiniLM‑L6‑v2`.
- **Indexing** → build a FAISS `IndexFlatIP` (inner product with normalized vectors ≈ cosine similarity) and persist it to disk.
- **Retrieval** → encode the user query, search top‑k, and assemble a context window from the retrieved texts.
- **Generation** → call an LLM via the OpenAI SDK using only the retrieved context; append a `Sources:` line with filenames.
- **UI** → run the full RAG loop in a Gradio app for quick evaluation by non‑technical users.

---

## Tools & libraries
- **Python 3.10+**
- **Sentence‑Transformers** (`all‑MiniLM‑L6‑v2`) for embeddings (Hugging Face ecosystem)
- **FAISS** (`faiss-cpu`) for vector search (cosine via normalized inner product)
- **OpenAI Python SDK** for LLM inference (chat completions)
- **Gradio 4** for the web UI
- **NumPy**, **python‑dotenv**, **Jupyter** for glue and notebooks

No keys are hard‑coded; credentials are loaded from `.env`.

---

## Repository layout (reference)
```
.
├── notebooks/
│   ├── 1.1_LoadAndEmbedDocs.ipynb        # build normalized embeddings -> data/processed/embed.npy
│   ├── 2.1_BuildFAISSIndex.ipynb         # build FAISS index -> data/processed/faiss_doc_index.index
│   ├── 3.1_Retrieval_DryRun.ipynb        # retrieval-only sanity checks
│   ├── 4.1_RetrieveAndGenerate.ipynb     # RAG single-shot (retrieve + LLM)
│   └── 5.1_GradioRAGApp.ipynb            # interactive UI
├── data/
│   ├── raw/                              # source docs referenced by docs_list.txt
│   │   ├── clinical_demo.md
│   │   └── docs_list.txt                 # filenames only, one per line
│   └── processed/
│       ├── embed.npy
│       └── faiss_doc_index.index
├── src/
│   ├── paths.py                          # canonical, portable paths (no hard-coded directories)
│   └── utils.py                          # helpers (optional)
├── .env                                  # contains OPENAI_API_KEY=...
├── requirements.txt
└── README.md
```
> Portable paths via `src/paths.py` so the same notebooks work locally and in CI.

---

## Setup

### 1) Create environment
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

> If `sentence-transformers` fails to resolve a compatible PyTorch wheel on your platform, install PyTorch per the instructions at https://pytorch.org and re-run `pip install -r requirements.txt`.

### 2) Configure credentials
Create a file named **`.env`** at the repository root:
```
OPENAI_API_KEY=sk-...
```
No quotes; one key per line.

### 3) Prepare documents
Place source files in `data/raw/` and list their **filenames** (not text) in `data/raw/docs_list.txt`, e.g.:
```
clinical_demo.md
```
`docs_list.txt` must contain only filenames that exist under `data/raw/`.

---

## How to run (notebooks)

1. **1.1_LoadAndEmbedDocs.ipynb**  
   Reads filenames from `docs_list.txt`, loads files from `data/raw/`, creates **normalized** embeddings with `all‑MiniLM‑L6‑v2`, and saves `data/processed/embed.npy`.  
   Expected: `Saved .../embed.npy (N, 384)` where `N` = number of files.

2. **2.1_BuildFAISSIndex.ipynb**  
   Loads `embed.npy`, builds `IndexFlatIP`, and saves `data/processed/faiss_doc_index.index`.  
   Expected: `ntotal == N`.

3. **3.1_Retrieval_DryRun.ipynb** (optional)  
   Encodes a sample query, searches FAISS, prints the top hit filenames, and previews context.

4. **4.1_RetrieveAndGenerate.ipynb**  
   Retrieves top‑k passages and calls the LLM with the retrieved context. Returns an answer plus `Sources: <filenames>`.

5. **5.1_GradioRAGApp.ipynb**  
   Launches a simple web UI to interact with the RAG system end‑to‑end.

---

## How to run (UI only)
Run **1.1** then **2.1** once to produce `embed.npy` and `faiss_doc_index.index`, then open **5.1_GradioRAGApp.ipynb** and run all cells. The app prints a local URL.

---

## Retrieval correctness
- Embeddings are **L2‑normalized** at creation time. With normalized vectors, **inner product** in FAISS is equivalent to **cosine similarity**.
- Query vectors are normalized at inference time for the same reason.
- Matching normalization on both sides is critical for stable nearest-neighbor behavior.

---

## Extending
- **Add docs**: place files under `data/raw/`, list them in `docs_list.txt`, rebuild **1.1** and **2.1**.
- **Swap encoders**: replace `all‑MiniLM‑L6‑v2`; keep normalization.
- **Change LLM**: modify the model name in generation cells or align to your provider’s SDK.
- **Chunking**: for long files, chunk in 1.1 before embedding to improve recall.

---

## Troubleshooting
- `OPENAI_API_KEY not found` → ensure `.env` is at repo root and restart the kernel/terminal.
- `Missing source file: …` → `docs_list.txt` must contain **filenames** that exist under `data/raw/`.
- `list index out of range` in 5.1 → artifacts mismatch (e.g., embeddings for 5 docs but `docs_list.txt` has 1). Re-run **1.1** then **2.1**.
- “Context not included” answers → verify 5.1 builds prompts from retrieved `docs_text` and appends `Sources:`.
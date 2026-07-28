"""Central configuration.

Everything that might change lives here so the rest of the code never
hard-codes a path or a model name. Freezing model choices early is a
deliberate anti-time-sink decision.
"""
from pathlib import Path

# ---- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"                 # put PDFs here
PROCESSED = DATA / "processed"     # chunked JSONL goes here
EVAL = DATA / "evaluation"         # benchmark questions + reference answers
RESULTS = ROOT / "results"

# ---- Chunking -------------------------------------------------------------
CHUNK_SIZE_WORDS = 220     # ~1 paragraph; big enough to be self-contained
CHUNK_OVERLAP_WORDS = 40   # overlap preserves context across boundaries

# ---- Models (frozen — do not shop around mid-project) ---------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"   # for claim verification later

# ---- Generation (API path) ------------------------------------------------
GEN_MODEL = "claude-sonnet-4-6"
GEN_MAX_TOKENS = 1024

# ---- Retrieval ------------------------------------------------------------
TOP_K = 5   # passages passed to the generator

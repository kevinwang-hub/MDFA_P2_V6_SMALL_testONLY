"""
Central configuration for the MOF Image Extraction Pipeline.
Override via environment variables.

MODEL SERVING:
Models are served via Ollama with OpenAI-compatible API on port 11434.
Available: qwen2.5vl:7b, gemma3:27b, qwen2.5:32b-instruct
"""

import os

# Model endpoints (all served via Ollama on same port)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
QWEN_VL_ENDPOINT = os.getenv("QWEN_VL_ENDPOINT", OLLAMA_BASE_URL)
GEMMA_VL_ENDPOINT = os.getenv("GEMMA_VL_ENDPOINT", OLLAMA_BASE_URL)
QWEN_TEXT_ENDPOINT = os.getenv("QWEN_TEXT_ENDPOINT", OLLAMA_BASE_URL)

# Model names (as registered in Ollama)
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen2.5vl:7b")
GEMMA_VL_MODEL = os.getenv("GEMMA_VL_MODEL", "gemma3:27b")
QWEN_TEXT_MODEL = os.getenv("QWEN_TEXT_MODEL", "qwen2.5:32b-instruct")

# Retrieval settings
BM25_TOP_K = 5
CONTEXT_TOKEN_BUDGET = 3000  # approximate, 4 chars ≈ 1 token

# Extraction settings
EXTRACTION_TEMPERATURE = 0.1
EXTRACTION_MAX_TOKENS = 4096
VERIFICATION_MAX_TOKENS = 6144  # larger because it includes corrected extraction
AGGREGATION_MAX_TOKENS = 8192

# Image processing
MAX_IMAGE_DIMENSION = 2048  # resize if larger (preserve aspect ratio)
SUPPORTED_IMAGE_FORMATS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]

# Pipeline settings
SKIP_LOW_RELEVANCE_THRESHOLD = 1  # skip extraction if relevance_to_synthesis <= this
SAVE_INTERMEDIATE = True  # save outputs from each phase
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

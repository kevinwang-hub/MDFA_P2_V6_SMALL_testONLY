# MOF Image Extraction Pipeline

A multi-phase pipeline for extracting structured scientific data from images in Metal-Organic Framework (MOF), Covalent Organic Framework (COF), and Zeolitic Imidazolate Framework (ZIF) research papers.

The pipeline uses vision-language models served via [Ollama](https://ollama.com/) to classify images, assemble paper context, extract domain-specific data, verify accuracy, and synthesize paper-level summaries.

## Architecture

The pipeline runs in 6 sequential phases:

| Phase | Module | Description |
|-------|--------|-------------|
| **0 — Context** | `phase0_context/` | Load parsed paper content, chunk text by section, build BM25 search index |
| **1 — Scout** | `phase1_scout/` | Classify each image (type, relevance, description) using Qwen 2.5 VL |
| **2 — Context Assembly** | `phase2_context/` | Match captions, retrieve relevant text chunks via BM25, build token-budgeted context |
| **3 — Extraction** | `phase3_extractor/` | Dispatch to type-specific prompts, extract structured JSON via Gemma 3 27B |
| **4 — Critic** | `phase4_critic/` | Adversarial verification — check extraction against image, flag errors, produce corrections |
| **5 — Synthesis** | `phase5_synthesizer/` | Aggregate all image extractions into a paper-level summary via Qwen 2.5 32B |

### Image Type Prompts

Phase 3 uses 10 specialized extraction prompts tailored to different image types:

- `adsorption` — Gas adsorption isotherms, BET surface area, pore size distribution
- `crystal_structure` — Ball-and-stick, polyhedral, topology diagrams
- `diffraction` — PXRD, single-crystal XRD patterns
- `spectroscopy` — FTIR, NMR, UV-Vis, XPS, Raman
- `thermal_analysis` — TGA, DSC, DTA curves
- `microscopy` — SEM, TEM, AFM morphology images
- `table_figure` — Tabular data with full cell extraction
- `synthesis_scheme` — Reaction diagrams with conditions
- `computational` — DFT, MD, band structure, DOS plots
- `generic` — Fallback for unclassified images

All prompts extract:
- **`text_in_image`** — Every visible text label (with anti-hallucination guard: "List ONLY text you can clearly read. Do NOT complete sequences, infer missing items, or extrapolate patterns.")
- **`diagram_representations`** — Visual encoding: color mapping, shape/marker mapping, line styles, notations/symbols

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with the following models pulled:
  - `qwen2.5vl:7b` — classifier (Phase 1)
  - `gemma3:27b` — extractor + critic (Phases 3–4)
  - `qwen2.5:32b-instruct` — synthesizer (Phase 5)

## Installation

```bash
pip3 install -r requirements.txt
```

```
rank-bm25>=0.2.2
Pillow>=10.0
requests>=2.31
openai>=1.0
tqdm>=4.65
numpy>=1.24
```

Pull the Ollama models:

```bash
ollama pull qwen2.5vl:7b
ollama pull gemma3:27b
ollama pull qwen2.5:32b-instruct
```

## Usage

### Full pipeline (all images in a paper)

```bash
python3 main.py \
  --paper /path/to/paper/hybrid_auto/ \
  --output output.json \
  --verbose
```

### Limit number of images

```bash
python3 main.py \
  --paper /path/to/paper/hybrid_auto/ \
  --max-images 3 \
  --output output.json
```

### Single image extraction

```bash
python3 run_single_image.py <image_filename> <output_path>
```

Example:

```bash
python3 run_single_image.py \
  db701208d417c1d16f203c8e35efb275eff8ef6877f11cf393c53bf7dc7cf1a4.jpg \
  ~/Downloads/result.json
```

> **Note:** `run_single_image.py` expects cached Phase 0 context from a prior full pipeline run on the same paper. Edit the `PAPER_DIR` variable inside the script to point to your paper directory.

### CLI Options

| Flag | Description |
|------|-------------|
| `--paper` | Path to the paper directory (must contain `content_list_v2.json` and `images/`) |
| `--output` | Output JSON file path |
| `--max-images` | Maximum number of images to process |
| `--image` | Process a single specific image filename |
| `--skip-phases` | Comma-separated phase numbers to skip (e.g., `0,1,2,3,4`) |
| `--verbose` | Enable verbose logging |

## Input Format

The pipeline expects a paper directory with:

```
paper_directory/
├── content_list_v2.json    # Parsed paper content (page-based nested format)
└── images/
    ├── <hash1>.jpg
    ├── <hash2>.jpg
    └── ...
```

`content_list_v2.json` is a nested list of pages, where each page is a list of elements:

```json
[
  [
    {"type": "text", "content": "...", "bbox": [...]},
    {"type": "image", "content": "<hash>.jpg", "bbox": [...]}
  ],
  ...
]
```

## Output Format

The output JSON contains:

```json
{
  "classification": { ... },
  "extraction": {
    "text_in_image": ["..."],
    "diagram_representations": {
      "color_mapping": [{"color": "blue", "represents": "..."}],
      "shape_mapping": [{"shape": "circle", "represents": "..."}],
      "notations_and_symbols": ["..."]
    },
    ...
  },
  "verification": { ... }
}
```

## Configuration

All settings are in `config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `QWEN_VL_MODEL` | `qwen2.5vl:7b` | Classifier model |
| `GEMMA_VL_MODEL` | `gemma3:27b` | Extractor/critic model |
| `QWEN_TEXT_MODEL` | `qwen2.5:32b-instruct` | Synthesizer model |
| `LOG_LEVEL` | `INFO` | Logging level |

## Project Structure

```
mof_pipeline/
├── config.py                          # Central configuration
├── main.py                            # CLI entry point
├── pipeline.py                        # Pipeline orchestrator
├── run_single_image.py                # Single image helper
├── requirements.txt
├── models/
│   ├── base.py                        # Base client class
│   ├── gemma_vl.py                    # Gemma 3 27B VL client
│   ├── qwen_vl.py                     # Qwen 2.5 VL client
│   └── qwen_text.py                   # Qwen 2.5 text-only client
├── phase0_context/
│   ├── loader.py                      # Paper content loader
│   ├── chunker.py                     # Text chunker
│   └── retriever.py                   # BM25 retrieval
├── phase1_scout/
│   ├── classifier.py                  # Image classifier
│   └── router.py                      # Prompt router
├── phase2_context/
│   ├── caption_matcher.py             # Caption matching
│   └── context_assembler.py           # Context assembly
├── phase3_extractor/
│   ├── extractor.py                   # Extraction dispatcher
│   └── prompts/                       # 10 type-specific prompts
│       ├── adsorption.py
│       ├── computational.py
│       ├── crystal_structure.py
│       ├── diffraction.py
│       ├── generic.py
│       ├── microscopy.py
│       ├── spectroscopy.py
│       ├── synthesis_scheme.py
│       ├── table_figure.py
│       └── thermal_analysis.py
├── phase4_critic/
│   └── verifier.py                    # Adversarial verification
├── phase5_synthesizer/
│   └── aggregator.py                  # Paper-level synthesis
└── utils/
    ├── image_utils.py                 # Image loading/encoding
    └── io_utils.py                    # JSON parsing, logging setup
```

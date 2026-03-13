"""Paper-level aggregation of verified extractions via Qwen3.5 9B."""

import json
import logging
from pathlib import Path

from config import AGGREGATION_MAX_TOKENS, EXTRACTION_TEMPERATURE
from models.qwen_text import QwenTextClient
from utils.io_utils import parse_json_response

logger = logging.getLogger(__name__)

# ── System prompt (no /no_think — think=False is set in the API call) ──
SYSTEM_PROMPT = """\
You are an expert materials chemist who synthesizes information from multiple analytical \
techniques to build comprehensive records of MOF/COF/ZIF materials reported in research papers.

You receive verified extractions from multiple figures/tables in a single paper.
Your job is to:
1. Identify all distinct materials reported in the paper
2. For each material, compile all information from all figures into a unified record
3. Resolve any conflicts between different sources
4. Identify gaps in the data

This data will be used to train a language model to predict synthesis conditions.
Accuracy and completeness are critical.

Output ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
=== PAPER METADATA ===
Title: {title}
Authors: {authors}
DOI: {doi}
Journal: {journal}

=== VERIFIED EXTRACTIONS FROM ALL FIGURES ===
{all_extractions_json}

=== FULL PAPER CONTEXT (key sections) ===
{paper_context_summary}

Compile a unified paper-level record.
Respond with ONLY this JSON:

{{
  "paper_info": {{
    "title": "<...>",
    "doi": "<...>",
    "journal": "<...>",
    "year": null,
    "paper_type": "<synthesis_report | review | computational | methodology | other>"
  }},

  "materials_reported": [
    {{
      "material_id": "<unique ID within this paper, e.g., MOF-1, MOF-2>",
      "name": "<primary name>",
      "aliases": ["<other names used in paper>"],
      "formula": "<chemical formula>",
      "material_class": "<MOF | COF | ZIF | MOP | PCP | other>",
      "is_novel": null,

      "synthesis": {{
        "synthesis_type": "<solvothermal | hydrothermal | mechanochemical | etc.>",
        "precursors": [
          {{
            "name": "<...>",
            "formula": "<...>",
            "role": "<metal_source | linker | modulator | etc.>",
            "amount": "<...>",
            "source_figure": "<which figure this came from>"
          }}
        ],
        "solvents": [
          {{"name": "<...>", "amount": "<...>", "source_figure": "<...>"}}
        ],
        "conditions": {{
          "temperature": "<°C>",
          "time": "<...>",
          "atmosphere": "<...>",
          "pH": "<...>",
          "vessel": "<...>"
        }},
        "post_synthesis": {{
          "washing": "<...>",
          "activation": "<...>",
          "solvent_exchange": "<...>"
        }},
        "yield": "<...>",
        "molar_ratios": {{
          "metal_to_linker": "<...>",
          "metal_to_modulator": "<...>"
        }}
      }},

      "structure": {{
        "topology": "<...>",
        "space_group": "<...>",
        "crystal_system": "<...>",
        "unit_cell": {{"a": "<Å>", "b": "<Å>", "c": "<Å>", "alpha": "<°>", "beta": "<°>", "gamma": "<°>"}},
        "SBU": "<...>",
        "linker": "<...>",
        "pore_dimensions": ["<...>"],
        "interpenetration": "<...>",
        "source_figures": ["<...>"]
      }},

      "properties": {{
        "BET_surface_area": {{"value": "<m²/g>", "source_figure": "<...>"}},
        "total_pore_volume": {{"value": "<cm³/g>", "source_figure": "<...>"}},
        "thermal_stability": {{"value": "<°C>", "source_figure": "<...>"}},
        "gas_uptake": [
          {{"gas": "<...>", "value": "<...>", "conditions": "<...>", "source_figure": "<...>"}}
        ],
        "selectivity": [
          {{"gas_pair": "<...>", "value": "<...>", "conditions": "<...>", "source_figure": "<...>"}}
        ],
        "crystallinity_confirmed": false,
        "characterization_techniques_used": ["<PXRD, FTIR, TGA, ...>"]
      }},

      "morphology": {{
        "crystal_habit": "<...>",
        "particle_size": "<...>",
        "source_figure": "<...>"
      }},

      "figure_to_material_mapping": {{}}
    }}
  ],

  "cross_figure_conflicts": [
    {{
      "field": "<...>",
      "figure_A": {{"figure_id": "<...>", "value": "<...>"}},
      "figure_B": {{"figure_id": "<...>", "value": "<...>"}},
      "resolution": "<which value to trust and why>"
    }}
  ],

  "data_gaps": [
    {{
      "material": "<which material>",
      "missing_field": "<what's missing>",
      "importance_for_synthesis_prediction": "<critical | important | nice_to_have>",
      "note": "<why it matters>"
    }}
  ]
}}"""

# Maximum chars for paper_context_summary sent to the model (~1K tokens)
MAX_CONTEXT_CHARS = 4000
# Maximum chars for all_extractions_json sent to the model
MAX_EXTRACTIONS_CHARS = 20000

# Keys to keep when stripping extractions for the prompt
_KEEP_KEYS = {
    "image_id", "image_path", "classification", "primary_type",
    "figure_id", "extracted_data", "verified_data", "corrected_extraction",
    "key_findings", "materials", "synthesis_conditions", "measurements",
}


def _slim_extraction(ext: dict) -> dict:
    """Strip an extraction dict to only prompt-relevant keys."""
    slim = {}
    for k, v in ext.items():
        if k.startswith("_"):
            continue
        if k in _KEEP_KEYS or not isinstance(v, (dict, list)):
            slim[k] = v
    # Flatten: prefer verified_data > corrected_extraction > extracted_data
    data = slim.pop("verified_data", None) or slim.pop("corrected_extraction", None) or slim.pop("extracted_data", None)
    if data:
        slim["data"] = data
    return slim


def fallback_synthesis(extractions: list[dict], paper_metadata: dict) -> dict:
    """Assemble valid output from Phase 4 results without LLM."""
    figures = []
    for ext in extractions:
        figures.append({
            "image_id": ext.get("_image_path", ext.get("image_id", "")),
            "classification": ext.get("primary_type", ext.get("classification", "")),
            "extracted_data": ext.get("verified_data",
                              ext.get("corrected_extraction",
                              ext.get("extracted_data", {}))),
        })
    return {
        "_fallback": True,
        "paper_info": {
            "title": paper_metadata.get("title", ""),
            "doi": paper_metadata.get("doi", ""),
            "journal": paper_metadata.get("journal", ""),
            "year": None,
            "paper_type": None,
        },
        "materials_reported": [],
        "figures": figures,
        "cross_figure_conflicts": [],
        "data_gaps": [],
    }


def _is_mostly_null(result: dict) -> bool:
    """Return True if the result is essentially empty / all nulls."""
    if not result or result.get("_parse_error"):
        return True
    materials = result.get("materials_reported", [])
    if not materials:
        return False  # empty materials list is still valid (might be correct)
    # Check if every material has no real data
    null_count = 0
    total = 0
    for mat in materials:
        for k, v in mat.items():
            total += 1
            if v is None or v == "" or v == [] or v == {}:
                null_count += 1
    if total > 0 and null_count / total > 0.8:
        return True
    return False


class Aggregator:
    """Aggregate verified extractions into a paper-level record."""

    def __init__(self, client: QwenTextClient | None = None):
        self.client = client or QwenTextClient()

    def aggregate(
        self,
        verified_extractions: list[dict],
        paper_metadata: dict,
        paper_context_summary: str,
    ) -> dict:
        """
        Compile all verified extractions into a unified paper record.
        Falls back to programmatic assembly if the LLM fails.
        """
        logger.info(
            "Aggregating %d extractions for: %s",
            len(verified_extractions),
            paper_metadata.get("title", "unknown"),
        )

        # ── Strip extractions to prompt-relevant data only ──
        clean_extractions = [_slim_extraction(ext) for ext in verified_extractions]
        all_extractions_json = json.dumps(clean_extractions, indent=1, ensure_ascii=False)
        # Truncate if still too big
        if len(all_extractions_json) > MAX_EXTRACTIONS_CHARS:
            all_extractions_json = all_extractions_json[:MAX_EXTRACTIONS_CHARS] + "\n... (truncated)"

        # ── Cap context summary ──
        ctx = paper_context_summary
        if len(ctx) > MAX_CONTEXT_CHARS:
            ctx = ctx[:MAX_CONTEXT_CHARS] + "\n... (truncated)"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=paper_metadata.get("title", ""),
            authors=paper_metadata.get("authors", ""),
            doi=paper_metadata.get("doi", ""),
            journal=paper_metadata.get("journal", ""),
            all_extractions_json=all_extractions_json,
            paper_context_summary=ctx,
        )

        # ── Debug: prompt size ──
        full_prompt = SYSTEM_PROMPT + user_prompt
        print(f"[Phase5 DEBUG] Prompt: {len(full_prompt)} chars / ~{len(full_prompt)//4} tokens")
        logger.info("[Phase5] Prompt size: %d chars / ~%d tokens", len(full_prompt), len(full_prompt) // 4)

        try:
            response = self.client.generate(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                temperature=EXTRACTION_TEMPERATURE,
                max_tokens=AGGREGATION_MAX_TOKENS,
            )

            logger.info("[Phase5] Raw response length: %d chars", len(response))
            result = parse_json_response(response)

            # Retry once if parsing failed
            if result.get("_parse_error"):
                logger.warning("Retrying aggregation (JSON parse failure)")
                retry_prompt = (
                    user_prompt
                    + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                    "Output ONLY the JSON object, no markdown, no explanation."
                )
                response = self.client.generate(
                    system=SYSTEM_PROMPT,
                    user=retry_prompt,
                    temperature=EXTRACTION_TEMPERATURE,
                    max_tokens=AGGREGATION_MAX_TOKENS,
                )
                result = parse_json_response(response)

            # Check if result is effectively empty
            if _is_mostly_null(result):
                logger.warning("[Phase5] LLM output is mostly null — using fallback")
                return fallback_synthesis(verified_extractions, paper_metadata)

            return result

        except Exception:
            logger.exception("[Phase5] LLM aggregation failed — using fallback")
            return fallback_synthesis(verified_extractions, paper_metadata)

"""Paper-level aggregation of verified extractions via Qwen2.5-32B-Instruct."""

import json
import logging

from config import AGGREGATION_MAX_TOKENS, EXTRACTION_TEMPERATURE
from models.qwen_text import QwenTextClient
from utils.io_utils import parse_json_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert materials chemist who synthesizes information from multiple analytical
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

        Args:
            verified_extractions: List of Phase 4 verified extraction dicts.
            paper_metadata: Dict with title, authors, doi, journal.
            paper_context_summary: Concatenated key paper sections.

        Returns:
            Paper-level aggregation dict.
        """
        logger.info(
            "Aggregating %d extractions for: %s",
            len(verified_extractions),
            paper_metadata.get("title", "unknown"),
        )

        # Strip internal metadata from extractions before sending to model
        clean_extractions = []
        for ext in verified_extractions:
            clean = {k: v for k, v in ext.items() if not k.startswith("_")}
            clean_extractions.append(clean)

        all_extractions_json = json.dumps(
            clean_extractions, indent=2, ensure_ascii=False
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=paper_metadata.get("title", ""),
            authors=paper_metadata.get("authors", ""),
            doi=paper_metadata.get("doi", ""),
            journal=paper_metadata.get("journal", ""),
            all_extractions_json=all_extractions_json,
            paper_context_summary=paper_context_summary,
        )

        response = self.client.generate(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=AGGREGATION_MAX_TOKENS,
        )

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

        return result

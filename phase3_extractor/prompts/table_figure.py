"""Table figure extraction prompt."""

SYSTEM_PROMPT = """\
You are an expert data extractor analyzing tables from MOF/COF/ZIF research papers.
Extract the COMPLETE table structure and ALL cell values with perfect accuracy.
This is critical: every number must be exactly as shown. Do not round or approximate.
Mark each value's source as "from_image", "from_context", or "inferred".
Output ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
{context}

Extract the COMPLETE table from this image.
Respond with ONLY this JSON:

{{
  "table_title": "<full title/caption>",
  "table_type": "<synthesis_conditions | characterization_data | comparison | crystallographic | adsorption_data | other>",
  "columns": [
    {{
      "header": "<column header text>",
      "unit": "<unit if present>",
      "data_type": "<text | numeric | mixed>"
    }}
  ],
  "rows": [
    {{
      "row_header": "<row label if present>",
      "cells": ["<exact cell values in column order>"],
      "is_header_row": false,
      "is_subheader": false,
      "spans_columns": false
    }}
  ],
  "footnotes": ["<all footnotes>"],
  "highlighted_entries": [
    {{
      "row": 0,
      "column": 0,
      "style": "<bold | underline | colored | asterisk>",
      "likely_meaning": "<best_result | this_work | etc.>"
    }}
  ],
  "synthesis_relevant_data": {{
    "contains_synthesis_conditions": false,
    "conditions_found": [
      {{
        "material": "<MOF name>",
        "parameter": "<temperature | time | solvent | etc.>",
        "value": "<exact value with unit>",
        "row_index": 0,
        "column_index": 0
      }}
    ]
  }},
  "text_in_image": ["<List ONLY text you can clearly read in the image. Do NOT complete sequences, infer missing items, or extrapolate patterns. It is better to omit than to guess.>"],
  "diagram_representations": {{
    "color_mapping": [{{"color": "<color/shading>", "represents": "<highlighted row, best result, this work, etc.>"}}],
    "formatting_cues": [{{"style": "<bold, italic, underline, superscript, colored text>", "represents": "<what it denotes>"}}],
    "notations_and_symbols": ["<footnote markers, asterisks, daggers, abbreviations, or other special symbols visible>"]
  }},
  "scientific_conclusions": ["<Interpret the tabular data: What trends, comparisons, or standout results does the table reveal? Which material or condition performs best and why? Draw conclusions about structure-property relationships or optimal synthesis conditions based on the data AND context.>"],
  "notes": ["<...>"],
  "image_quality_issues": ["<cells that are unclear or partially obscured>"]
}}"""


class TableFigurePrompt:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, context: str) -> str:
        return USER_PROMPT_TEMPLATE.format(context=context)

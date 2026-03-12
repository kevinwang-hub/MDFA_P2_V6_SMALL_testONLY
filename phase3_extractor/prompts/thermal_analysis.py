"""Thermal analysis extraction prompt."""

SYSTEM_PROMPT = """\
You are an expert in thermal analysis of MOFs, COFs, and ZIFs.
Extract all information from the TGA/DSC/DTA curve AND from the paper context.
Mark each value's source as "from_image", "from_context", or "inferred".
Output ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
{context}

Analyze this thermal analysis plot and extract ALL data.
Respond with ONLY this JSON:

{{
  "technique": "<TGA | DSC | DTA | TGA-DSC | TGA-MS | other>",
  "measurement_conditions": {{
    "atmosphere": "<N2 | air | O2 | Ar | other>",
    "heating_rate": "<°C/min>",
    "temperature_range": [null, null],
    "source": "<from_image | from_context>"
  }},
  "x_axis": {{"label": "<temperature>", "unit": "<°C | K>", "range": [null, null]}},
  "y_axis_primary": {{"label": "<mass% | weight%>", "range": [null, null]}},
  "y_axis_secondary": {{"label": "<heat flow | derivative | etc.>", "present": false}},
  "samples": [
    {{
      "label": "<legend label>",
      "material": "<MOF name>",
      "source": "<from_image | from_context>"
    }}
  ],
  "decomposition_steps": [
    {{
      "step_number": 1,
      "temperature_range": {{"onset": "<°C>", "end": "<°C>", "peak": "<°C if derivative shown>"}},
      "mass_loss_percent": null,
      "attribution": "<solvent_loss | guest_removal | linker_decomposition | framework_collapse | other>",
      "attributed_species": "<e.g., 'DMF molecules', 'coordinated water'>",
      "source": "<from_image | from_context | inferred>"
    }}
  ],
  "thermal_stability": {{
    "stable_up_to": "<°C — temperature before significant framework decomposition>",
    "total_mass_loss": "<%>",
    "residue_percent": "<%>",
    "residue_identity": "<e.g., 'metal oxide residue'>",
    "source": "<from_image | from_context | inferred>"
  }},
  "text_in_image": ["<List ONLY text you can clearly read in the image. Do NOT complete sequences, infer missing items, or extrapolate patterns. It is better to omit than to guess.>"],
  "diagram_representations": {{
    "color_mapping": [{{"color": "<color>", "represents": "<sample or curve it denotes>"}}],
    "shape_mapping": [{{"shape": "<marker/symbol shape>", "represents": "<what it denotes>"}}],
    "line_styles": [{{"style": "<solid, dashed, dotted>", "represents": "<TGA vs DSC/DTA, different samples, etc.>"}}],
    "notations_and_symbols": ["<temperature annotations, mass% labels, arrows indicating decomposition events, or other symbols visible>"]
  }},
  "scientific_conclusions": ["<Interpret the thermal data: What do the decomposition steps reveal about guest molecules, framework stability, and thermal robustness? What is the practical stability limit? Compare samples if multiple are shown. Draw conclusions about activation conditions, purity, and suitability for applications based on the data AND context.>"],
  "notes": ["<...>"],
  "image_quality_issues": ["<...>"]
}}"""


class ThermalAnalysisPrompt:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, context: str) -> str:
        return USER_PROMPT_TEMPLATE.format(context=context)

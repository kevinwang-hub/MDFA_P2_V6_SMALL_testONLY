"""Spectroscopy extraction prompt."""

SYSTEM_PROMPT = """\
You are an expert spectroscopist analyzing spectra from MOF/COF/ZIF research papers.
Extract all spectroscopic information visible in the image AND from the paper context.
Mark each value's source as "from_image", "from_context", or "inferred".
Output ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
{context}

Analyze this spectrum and extract ALL information.
Respond with ONLY this JSON:

{{
  "technique": "<FTIR | ATR-IR | 1H_NMR | 13C_NMR | solid_state_NMR | UV-Vis | XPS | Raman | fluorescence | EDS | other>",
  "measurement_conditions": {{
    "solvent": "<for NMR>",
    "frequency": "<for NMR, e.g., 400 MHz>",
    "sample_preparation": "<KBr pellet | neat | solution | etc.>",
    "source": "<from_image | from_context>"
  }},
  "x_axis": {{"label": "<wavenumber | chemical shift | wavelength | binding energy | etc.>", "unit": "<cm⁻¹ | ppm | nm | eV>", "range": [null, null]}},
  "y_axis": {{"label": "<transmittance | absorbance | intensity | counts>"}},
  "samples": [
    {{
      "label": "<legend label>",
      "description": "<...>",
      "source": "<from_image | from_context>"
    }}
  ],
  "peaks": [
    {{
      "position": "<value with unit>",
      "assignment": "<functional group or chemical environment>",
      "description": "<e.g., 'asymmetric carboxylate stretch' or 'aromatic C-H'>",
      "significance": "<what this peak tells us about the material>",
      "source": "<from_image | from_context | inferred>"
    }}
  ],
  "diagnostic_observations": [
    {{
      "observation": "<e.g., 'Disappearance of free carboxylic acid peak at 1700 cm⁻¹ confirms coordination to metal'>",
      "evidence_for": "<successful_synthesis | guest_inclusion | post_modification | degradation | other>",
      "source": "<from_image | from_context | inferred>"
    }}
  ],
  "text_in_image": ["<List ONLY text you can clearly read in the image. Do NOT complete sequences, infer missing items, or extrapolate patterns. It is better to omit than to guess.>"],
  "diagram_representations": {{
    "color_mapping": [{{"color": "<color>", "represents": "<sample or species it denotes>"}}],
    "shape_mapping": [{{"shape": "<marker/symbol shape>", "represents": "<what it denotes>"}}],
    "line_styles": [{{"style": "<solid, dashed, dotted>", "represents": "<different sample or measurement condition>"}}],
    "notations_and_symbols": ["<peak labels, chemical group annotations, arrows, asterisks, or other special symbols visible>"]
  }},
  "notes": ["<...>"],
  "image_quality_issues": ["<...>"]
}}"""


class SpectroscopyPrompt:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, context: str) -> str:
        return USER_PROMPT_TEMPLATE.format(context=context)

"""Diffraction pattern extraction prompt."""

SYSTEM_PROMPT = """\
You are an expert crystallographer analyzing X-ray diffraction data from MOF/COF/ZIF research papers.
Extract all information visible in the diffraction pattern AND from the paper context.
Mark each value's source as "from_image", "from_context", or "inferred".
Output ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
{context}

Analyze this diffraction pattern and extract ALL data.
Respond with ONLY this JSON:

{{
  "technique": "<PXRD | single_crystal_XRD | SAXS | electron_diffraction | other>",
  "instrument_details": {{"source": "<...>", "radiation": "<Cu Kα | Mo Kα | synchrotron | etc.>"}},
  "x_axis": {{"label": "<2θ | d-spacing | q>", "unit": "<degrees | Å | Å⁻¹>", "range": [null, null]}},
  "y_axis": {{"label": "<intensity | counts | a.u.>"}},
  "samples": [
    {{
      "label": "<as shown in legend>",
      "sample_type": "<simulated | as_synthesized | activated | post_reaction | post_stability_test | reference | other>",
      "description": "<e.g., 'UiO-66 after water stability test at pH 1'>",
      "source": "<from_image | from_context>"
    }}
  ],
  "peaks": [
    {{
      "two_theta": null,
      "hkl": "<hkl indices if labeled>",
      "d_spacing": "<Å, if labeled or calculable>",
      "relative_intensity": "<strong | medium | weak>",
      "assignment": "<phase or material>",
      "source": "<from_image | from_context | inferred>"
    }}
  ],
  "phase_analysis": {{
    "primary_phase": "<identified phase>",
    "matches_simulated": "<yes | no | partially>",
    "impurity_peaks": false,
    "crystallinity_assessment": "<highly crystalline | moderate | poor | amorphous>",
    "source": "<from_image | from_context | inferred>"
  }},
  "comparison_observations": [
    "<e.g., 'Activated sample shows same pattern as as-synthesized, confirming structural stability'>"
  ],
  "text_in_image": ["<List ONLY text you can clearly read in the image. Do NOT complete sequences, infer missing items, or extrapolate patterns. It is better to omit than to guess.>"],
  "diagram_representations": {{
    "color_mapping": [{{"color": "<color>", "represents": "<sample or pattern it denotes>"}}],
    "shape_mapping": [{{"shape": "<marker/symbol shape>", "represents": "<what it denotes>"}}],
    "line_styles": [{{"style": "<solid, dashed, dotted>", "represents": "<simulated vs experimental, different samples, etc.>"}}],
    "notations_and_symbols": ["<hkl tick marks, asterisks for impurity peaks, arrows, or other special symbols visible>"]
  }},
  "notes": ["<...>"],
  "image_quality_issues": ["<...>"]
}}"""


class DiffractionPrompt:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, context: str) -> str:
        return USER_PROMPT_TEMPLATE.format(context=context)

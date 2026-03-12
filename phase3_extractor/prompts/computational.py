"""Computational figure extraction prompt."""

SYSTEM_PROMPT = """\
You are an expert computational chemist analyzing simulation and calculation results
from MOF/COF/ZIF research papers.
Extract all computational information from the image AND from the paper context.
Mark each value's source as "from_image", "from_context", or "inferred".
Output ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
{context}

Analyze this computational figure and extract ALL information.
Respond with ONLY this JSON:

{{
  "computation_type": "<DFT | MD | GCMC | ESP | band_structure | DOS | simulated_pattern | energy_diagram | other>",
  "software": "<VASP | Gaussian | CASTEP | LAMMPS | Materials Studio | etc. — if mentioned in context>",
  "method_details": {{
    "functional": "<PBE | B3LYP | etc.>",
    "basis_set": "<if mentioned>",
    "dispersion_correction": "<D3 | D3BJ | TS | none>",
    "source": "<from_context>"
  }},
  "visualization_content": {{
    "description": "<what is shown>",
    "color_scale": {{"present": false, "range": "<e.g., -0.05 to 0.05 e/Å³>", "meaning": "<charge density | potential | energy>"}},
    "isosurface_value": "<if shown>"
  }},
  "quantitative_results": [
    {{
      "property": "<band_gap | binding_energy | adsorption_energy | charge_transfer | etc.>",
      "value": "<with units>",
      "conditions": "<what system/configuration>",
      "source": "<from_image | from_context>"
    }}
  ],
  "comparison_data": [
    {{
      "what_is_compared": "<e.g., 'simulated vs experimental PXRD'>",
      "agreement": "<excellent | good | fair | poor>",
      "source": "<from_image | inferred>"
    }}
  ],
  "text_in_image": ["<List ONLY text you can clearly read in the image. Do NOT complete sequences, infer missing items, or extrapolate patterns. It is better to omit than to guess.>"],
  "diagram_representations": {{
    "color_mapping": [{{"color": "<color>", "represents": "<charge density, potential, orbital, energy level, etc.>"}}],
    "shape_mapping": [{{"shape": "<isosurface, arrow, sphere, lobe, etc.>", "represents": "<what it denotes>"}}],
    "notations_and_symbols": ["<energy values, axis labels, Fermi level markers, symmetry point labels, or other symbols visible>"]
  }},
  "notes": ["<...>"],
  "image_quality_issues": ["<...>"]
}}"""


class ComputationalPrompt:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, context: str) -> str:
        return USER_PROMPT_TEMPLATE.format(context=context)

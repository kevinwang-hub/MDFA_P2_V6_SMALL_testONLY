"""Microscopy image extraction prompt."""

SYSTEM_PROMPT = """\
You are an expert microscopist analyzing MOF/COF/ZIF crystal images.
Extract all morphological and microstructural information from the image AND from the paper context.
Mark each value's source as "from_image", "from_context", or "inferred".
Output ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
{context}

Analyze this microscopy image and extract ALL information.
Respond with ONLY this JSON:

{{
  "technique": "<SEM | TEM | HR-TEM | AFM | optical_microscopy | STEM | other>",
  "imaging_conditions": {{
    "accelerating_voltage": "<kV>",
    "magnification": "<e.g., 50,000x>",
    "detector": "<SE | BSE | HAADF | etc.>",
    "source": "<from_image | from_context>"
  }},
  "scale_bar": {{
    "present": false,
    "value": "<e.g., 1 μm, 200 nm>",
    "estimated_accuracy": "<reliable | hard_to_read>"
  }},
  "morphology": {{
    "crystal_habit": "<cubic | octahedral | rod | needle | plate | hexagonal | spherical | irregular | core_shell | other>",
    "description": "<detailed morphology description>",
    "uniformity": "<uniform | heterogeneous | bimodal>",
    "source": "<from_image | inferred>"
  }},
  "particle_size": {{
    "estimated_range": "<e.g., 200-500 nm>",
    "average": "<e.g., ~300 nm>",
    "measurement_method": "<from_scale_bar | from_context>",
    "source": "<from_image | from_context | inferred>"
  }},
  "surface_features": {{
    "texture": "<smooth | rough | porous | layered | faceted>",
    "defects": "<cracks | holes | surface_roughening | none_visible>",
    "source": "<from_image | inferred>"
  }},
  "agglomeration": "<none | mild | severe>",
  "panels": [
    {{
      "panel_label": "<a, b, c, etc.>",
      "description": "<what this specific panel shows>",
      "conditions": "<e.g., 'before activation', 'after cycling', 'sample B'>"
    }}
  ],
  "elemental_analysis_inset": {{
    "present": false,
    "type": "<EDS | EELS | elemental_mapping>",
    "elements_detected": ["<element symbols>"],
    "composition": {{}},
    "source": "<from_image>"
  }},
  "lattice_fringes": {{
    "visible": false,
    "d_spacing": "<Å>",
    "zone_axis": "<if labeled>",
    "source": "<from_image | inferred>"
  }},
  "text_in_image": ["<List ONLY text you can clearly read in the image. Do NOT complete sequences, infer missing items, or extrapolate patterns. It is better to omit than to guess.>"],
  "diagram_representations": {{
    "color_mapping": [{{"color": "<color>", "represents": "<element, phase, or feature it denotes in overlays/mappings>"}}],
    "shape_mapping": [{{"shape": "<marker/symbol/annotation shape>", "represents": "<what it denotes>"}}],
    "notations_and_symbols": ["<scale bars, panel labels, arrows, circles highlighting features, inset labels, or other annotations visible>"]
  }},
  "scientific_conclusions": ["<Interpret the morphology: What do crystal shape, size, uniformity, and surface features reveal about synthesis quality, growth conditions, or phase purity? Compare panels if multiple are shown (e.g., before/after treatment). Draw conclusions about crystallinity, defects, or structural integrity based on the data AND context.>"],
  "notes": ["<...>"],
  "image_quality_issues": ["<...>"]
}}"""


class MicroscopyPrompt:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, context: str) -> str:
        return USER_PROMPT_TEMPLATE.format(context=context)

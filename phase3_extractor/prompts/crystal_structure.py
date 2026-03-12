"""Crystal structure extraction prompt."""

SYSTEM_PROMPT = """\
You are an expert crystallographer analyzing crystal structure illustrations of MOFs, COFs, and ZIFs.
Extract all structural information from the visualization AND from the paper context.
Mark each value's source as "from_image", "from_context", or "inferred".
Output ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
{context}

Analyze this crystal structure image and extract ALL structural information.
Respond with ONLY this JSON:

{{
  "representation_type": "<ball_and_stick | polyhedral | space_filling | wireframe | topology_diagram | SBU_diagram | mixed>",
  "views_shown": ["<e.g., 'along a-axis', 'perspective view', '3D packing'>"],
  "framework_info": {{
    "name": "<MOF/COF/ZIF name>",
    "formula": "<if shown or in context>",
    "topology": "<e.g., pcu, fcu, dia, sql, etc.>",
    "dimensionality": "<2D | 3D>",
    "interpenetration": {{"present": false, "fold": "<e.g., 2-fold>"}},
    "source": "<from_image | from_context>"
  }},
  "metal_node_SBU": {{
    "metal_centers": ["<metal element symbols>"],
    "coordination_number": null,
    "coordination_geometry": "<octahedral | tetrahedral | paddlewheel | trinuclear | etc.>",
    "SBU_type": "<e.g., Zr6O4(OH)4, Cu2(COO)4, Zn4O>",
    "source": "<from_image | from_context | inferred>"
  }},
  "linkers": [
    {{
      "name": "<common name, e.g., BDC, BTC, BPDC>",
      "formula": "<chemical formula>",
      "type": "<dicarboxylate | tricarboxylate | imidazolate | pyridyl | mixed | other>",
      "source": "<from_image | from_context>"
    }}
  ],
  "pore_information": {{
    "pore_types": ["<e.g., 'large hexagonal channels', 'small tetrahedral cages'>"],
    "pore_dimensions": ["<e.g., '11.2 Å diameter'>"],
    "window_dimensions": ["<aperture sizes>"],
    "source": "<from_image | from_context | inferred>"
  }},
  "crystallographic_data": {{
    "space_group": "<e.g., Fm-3m>",
    "crystal_system": "<cubic | tetragonal | hexagonal | etc.>",
    "unit_cell": {{"a": "<Å>", "b": "<Å>", "c": "<Å>", "alpha": "<°>", "beta": "<°>", "gamma": "<°>"}},
    "source": "<from_context>"
  }},
  "text_in_image": ["<List ONLY text you can clearly read in the image. Do NOT complete sequences, infer missing items, or extrapolate patterns. It is better to omit than to guess.>"],
  "diagram_representations": {{
    "color_mapping": [{{"color": "<color>", "represents": "<atom type, element, or structural component>"}}],
    "shape_mapping": [{{"shape": "<sphere, polyhedron, stick, wireframe, etc.>", "represents": "<what atom/bond/unit it denotes>"}}],
    "notations_and_symbols": ["<axis labels, Miller indices, distance annotations, angle annotations, or other symbols visible>"]
  }},
  "notes": ["<...>"],
  "image_quality_issues": ["<...>"]
}}"""


class CrystalStructurePrompt:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, context: str) -> str:
        return USER_PROMPT_TEMPLATE.format(context=context)

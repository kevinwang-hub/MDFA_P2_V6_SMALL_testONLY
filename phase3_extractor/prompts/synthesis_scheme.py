"""Synthesis scheme extraction prompt."""

SYSTEM_PROMPT = """\
You are an expert materials chemist specializing in the synthesis of Metal-Organic Frameworks (MOFs),
Covalent Organic Frameworks (COFs), and Zeolitic Imidazolate Frameworks (ZIFs).

You are analyzing a synthesis scheme or reaction diagram from a research paper.
Your task is to extract EVERY detail about synthesis conditions visible in the image
AND supplement with information from the paper context provided.

For each piece of extracted data, mark its source as:
  "from_image" — directly visible/readable in the image
  "from_context" — obtained from the paper text provided
  "inferred" — logically inferred (explain reasoning)

Output ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
{context}

Analyze this synthesis scheme image and extract ALL synthesis information.
Respond with ONLY this JSON:

{{
  "synthesis_type": "<solvothermal | hydrothermal | mechanochemical | microwave | electrochemical | room_temperature | reflux | other>",
  "target_material": {{
    "name": "<MOF/COF/ZIF name>",
    "formula": "<chemical formula if shown>",
    "aliases": ["<alternative names>"]
  }},
  "steps": [
    {{
      "step_number": <int>,
      "step_type": "<synthesis | activation | washing | solvent_exchange | drying | other>",
      "description": "<brief description of this step>",
      "reagents": [
        {{
          "name": "<chemical name>",
          "formula": "<formula>",
          "role": "<metal_source | linker | modulator | solvent | base | acid | catalyst | other>",
          "amount": "<amount with units>",
          "molar_ratio": "<relative to metal, if determinable>",
          "source": "<from_image | from_context | inferred>"
        }}
      ],
      "solvents": [
        {{
          "name": "<solvent name>",
          "volume": "<volume with units>",
          "ratio": "<if mixed solvent, ratio>",
          "source": "<from_image | from_context | inferred>"
        }}
      ],
      "conditions": {{
        "temperature": {{"value": "<°C>", "ramp_rate": "<°C/min or null>", "source": "<...>"}},
        "time": {{"value": "<duration with units>", "source": "<...>"}},
        "pressure": {{"value": "<if mentioned>", "source": "<...>"}},
        "atmosphere": {{"value": "<air | N2 | Ar | vacuum | other>", "source": "<...>"}},
        "pH": {{"value": "<if mentioned>", "source": "<...>"}},
        "vessel": {{"value": "<Teflon-lined autoclave | round-bottom flask | etc.>", "source": "<...>"}},
        "cooling": {{"value": "<natural cooling | quench | ramp down at X°C/hr>", "source": "<...>"}}
      }}
    }}
  ],
  "post_synthesis": {{
    "washing": {{"solvents": ["<...>"], "method": "<...>", "source": "<...>"}},
    "solvent_exchange": {{"solvent": "<...>", "duration": "<...>", "cycles": "<int>", "source": "<...>"}},
    "activation": {{"method": "<vacuum | supercritical_CO2 | heat>", "temperature": "<...>", "duration": "<...>", "source": "<...>"}},
    "yield": {{"value": "<...>", "source": "<...>"}}
  }},
  "molar_ratios": {{
    "metal_to_linker": "<...>",
    "metal_to_modulator": "<...>",
    "source": "<...>"
  }},
  "text_in_image": ["<List ONLY text you can clearly read in the image. Do NOT complete sequences, infer missing items, or extrapolate patterns. It is better to omit than to guess.>"],
  "diagram_representations": {{
    "color_mapping": [{{"color": "<color>", "represents": "<reagent, product, solvent, or structural component it denotes>"}}],
    "shape_mapping": [{{"shape": "<icon, arrow type, box, or symbol shape>", "represents": "<reaction step, intermediate, or meaning>"}}],
    "notations_and_symbols": ["<reaction arrows, plus signs, temperature/time annotations, structural diagrams, abbreviations, or other symbols visible>"]
  }},
  "notes": ["<any additional observations, unusual conditions, or important details>"],
  "image_quality_issues": ["<any parts of the image that are unclear, cut off, or hard to read>"]
}}"""


class SynthesisSchemePrompt:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, context: str) -> str:
        return USER_PROMPT_TEMPLATE.format(context=context)
